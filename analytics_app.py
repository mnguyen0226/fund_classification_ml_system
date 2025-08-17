# analytics_app.py
from pathlib import Path
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
FDBK_DIR = ROOT / "feedback"
MODELS_DIR = ROOT / "models"
ANALYTICS_DIR = ROOT / "analytics"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="ML System Analytics", page_icon="📊", layout="wide")


# ---------------------- Helpers & Caching ----------------------
@st.cache_data(show_spinner=False)
def list_feedback_files() -> list[Path]:
    return sorted(FDBK_DIR.rglob("labels_*.csv"))


@st.cache_data(show_spinner=False)
def load_feedback_df() -> pd.DataFrame:
    files = list_feedback_files()
    if not files:
        return pd.DataFrame()
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            df["__file"] = str(p)
            frames.append(df)
        except Exception as e:
            # Skip unreadable files but continue
            print(f"[WARN] Failed to read {p}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Back-compat normalize
    if "predicted_label" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "predicted_label"})
    for col in (
        "correct_label",
        "timestamp_utc",
        "model_version",
        "user_id",
        "name",
        "predicted_label",
    ):
        if col not in df.columns:
            df[col] = None

    # Chosen/final for a feedback row
    df["chosen_label"] = df.apply(
        lambda r: (
            r["correct_label"]
            if pd.notna(r.get("correct_label"))
            and str(r["correct_label"]).strip() != ""
            else r["predicted_label"]
        ),
        axis=1,
    )
    df["disagree"] = df["chosen_label"] != df["predicted_label"]

    # Parse timestamps
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)

    # Fallback to folder date YYYY-MM-DD
    if ts.isna().any():

        def fallback(row):
            try:
                parts = Path(row["__file"]).parts
                for part in parts:
                    if len(part) == 10 and part[4] == "-" and part[7] == "-":
                        return pd.to_datetime(part, errors="coerce", utc=True)
            except Exception:
                pass
            return pd.NaT

        ts = ts.fillna(df.apply(fallback, axis=1))

    df["timestamp_utc"] = ts
    df["week"] = df["timestamp_utc"].dt.to_period("W").dt.start_time
    return df


@st.cache_data(show_spinner=False)
def load_train_log() -> pd.DataFrame:
    p = MODELS_DIR / "train_log.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, parse_dates=["timestamp_utc"])
    except Exception:
        # Fallback parse if formats are odd
        df = pd.read_csv(p)
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"], errors="coerce", utc=True
        )
    return df


def plot_line(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def normalize_tokens(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
        .str.split()
        .explode()
        .dropna()
    )


# ---------------------- Sidebar Filters ----------------------
with st.sidebar:
    st.header("Filters")

    fbk_df_full = load_feedback_df()
    if not fbk_df_full.empty:
        min_dt = fbk_df_full["timestamp_utc"].dropna().min()
        max_dt = fbk_df_full["timestamp_utc"].dropna().max()
    else:
        min_dt = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=30)
        max_dt = pd.Timestamp(datetime.now(timezone.utc))

    # Convert to dates for UI
    def _to_date(ts):
        if pd.isna(ts):
            return datetime.now(timezone.utc).date()
        return ts.date()

    date_range = st.date_input(
        "Feedback date range",
        value=(_to_date(min_dt), _to_date(max_dt)),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = _to_date(min_dt)
        end_date = _to_date(max_dt)

    username_filter = st.text_input("Username contains (optional)", value="").strip()
    label_filter = st.multiselect(
        "Chosen label filter",
        options=["Fund", "Non-Fund"],
        default=["Fund", "Non-Fund"],
    )

# ---------------------- Main UI ----------------------
st.title("📊 ML System Analytics")
st.caption("Covers feedback quality & model training performance.")

tab1, tab2 = st.tabs(["📝 Feedback Analytics", "🧪 Training Analytics"])

# ====================== TAB 1: Feedback Analytics ======================
with tab1:
    st.subheader("Feedback Overview")

    if fbk_df_full.empty:
        st.info(
            "No feedback found yet. Submit feedback in the main app to populate analytics."
        )
    else:
        # Apply sidebar filters
        fbk = fbk_df_full.copy()
        # Date range (inclusive end date by adding one day)
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        fbk = fbk[(fbk["timestamp_utc"] >= start_ts) & (fbk["timestamp_utc"] < end_ts)]

        if username_filter:
            fbk = fbk[
                fbk["user_id"]
                .astype(str)
                .str.contains(username_filter, case=False, na=False)
            ]

        if label_filter:
            fbk = fbk[fbk["chosen_label"].isin(label_filter)]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows (filtered)", len(fbk))
        with c2:
            st.metric("Users (filtered)", fbk["user_id"].nunique())
        with c3:
            st.metric(
                "Disagreement rate",
                f"{fbk['disagree'].mean():.2%}" if len(fbk) else "n/a",
            )
        with c4:
            st.metric("Model versions", fbk["model_version"].nunique())

        # Weekly disagreement
        st.markdown("### Weekly Disagreement Rate")
        weekly = (
            fbk.dropna(subset=["week"])
            .groupby("week")["disagree"]
            .mean()
            .reset_index()
            .sort_values("week")
        )
        if weekly.empty:
            st.info("No weekly data to plot for the current filters.")
        else:
            fig = plot_line(
                weekly["week"],
                weekly["disagree"],
                "Disagreement Rate by Week",
                "Week",
                "Disagreement (mean)",
            )
            st.pyplot(fig, use_container_width=True)

        # Per-user throughput & disagreement
        st.markdown("### Per-user Throughput & Disagreement")
        per_user = (
            fbk.groupby("user_id")
            .agg(rows=("name", "count"), disagree_rate=("disagree", "mean"))
            .reset_index()
            .sort_values("rows", ascending=False)
        )
        st.dataframe(per_user, use_container_width=True)

        # Disagreement by model version
        st.markdown("### Disagreement by Model Version")
        by_ver = (
            fbk.groupby("model_version")["disagree"]
            .mean()
            .reset_index()
            .sort_values("disagree", ascending=False)
        )
        st.dataframe(by_ver, use_container_width=True)

        # Weekly class counts (chosen_label)
        st.markdown("### Weekly Class Counts (Chosen Label)")
        wk_class = (
            fbk.dropna(subset=["week"])
            .groupby(["week", "chosen_label"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .sort_values("week")
        )
        st.dataframe(wk_class.tail(20), use_container_width=True)

        # Hot tokens among disagreements
        st.markdown("### Top Tokens in Disagreements")
        err = fbk[fbk["disagree"] == True]
        if not err.empty:
            tokens = normalize_tokens(err["name"])
            top_tok = tokens.value_counts().head(30).reset_index()
            top_tok.columns = ["token", "count"]
            st.dataframe(top_tok, use_container_width=True)
        else:
            st.info("No disagreements in the current filter selection.")

# ====================== TAB 2: Training Analytics ======================
with tab2:
    st.subheader("Model Training Performance")

    train_log = load_train_log()
    if train_log.empty:
        st.info(
            "No training logs found. Run `python train_model.py` to generate `models/train_log.csv`."
        )
    else:
        train_log = train_log.sort_values("timestamp_utc")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Retrains", len(train_log))
        with c2:
            st.metric("Latest accuracy", f"{train_log.iloc[-1]['acc']:.3f}")
        with c3:
            st.metric("Models logged", train_log["model_file"].nunique())

        st.markdown("### Accuracy Over Time")
        fig2 = plot_line(
            train_log["timestamp_utc"],
            train_log["acc"],
            "Model Accuracy Over Time",
            "Timestamp (UTC)",
            "Accuracy",
        )
        st.pyplot(fig2, use_container_width=True)

        st.markdown("### Latest 20 Retrains")
        st.dataframe(
            train_log.tail(20)[
                [
                    "timestamp_utc",
                    "version",
                    "model_file",
                    "acc",
                    "n_total",
                    "n_train",
                    "n_test",
                    "precision_Fund",
                    "recall_Fund",
                    "precision_NonFund",
                    "recall_NonFund",
                ]
            ],
            use_container_width=True,
        )

        # Quick filter by version substring
        ver_q = st.text_input("Filter rows by version contains", value="").strip()
        if ver_q:
            st.dataframe(
                train_log[
                    train_log["version"]
                    .astype(str)
                    .str.contains(ver_q, case=False, na=False)
                ],
                use_container_width=True,
            )

st.caption("This dashboard reads CSVs produced by your main app & training pipeline.")
