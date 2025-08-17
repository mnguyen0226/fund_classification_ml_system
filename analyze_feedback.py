# analyze_feedback.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FDBK = ROOT / "feedback"
OUTDIR = ROOT / "analytics"
OUTDIR.mkdir(exist_ok=True, parents=True)


def _list_feedback_files():
    files = sorted(FDBK.rglob("labels_*.csv"))
    print(f"Found {len(files)} feedback file(s) under {FDBK}")
    for p in files[:5]:
        print(f"  - {p}")
    if len(files) > 5:
        print("  ...")
    return files


def load_feedback() -> pd.DataFrame:
    files = _list_feedback_files()
    if not files:
        return pd.DataFrame()

    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            df["__file"] = str(p)
            frames.append(df)
        except Exception as e:
            print(f"WARNING: failed to read {p}: {e}")
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
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

    parsed = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    if parsed.isna().any():
        # fallback: derive from folder name YYYY-MM-DD if present
        def fallback(row):
            parts = Path(row["__file"]).parts
            for part in parts:
                if len(part) == 10 and part[4] == "-" and part[7] == "-":
                    return pd.to_datetime(part, errors="coerce", utc=True)
            return pd.NaT

        parsed = parsed.fillna(df.apply(fallback, axis=1))

    df["timestamp_utc"] = parsed
    print(
        f"Rows total: {len(df)} | with timestamp: {(~df['timestamp_utc'].isna()).sum()} | without: {df['timestamp_utc'].isna().sum()}"
    )
    df["week"] = df["timestamp_utc"].dt.to_period("W").dt.start_time
    return df


def plot_weekly_disagreement(df: pd.DataFrame):
    df = df.dropna(subset=["week"])
    weekly = df.groupby("week")["disagree"].mean().reset_index()
    if weekly.empty:
        print("No weekly data to plot.")
        return
    plt.figure()
    plt.plot(weekly["week"], weekly["disagree"], marker="o")
    plt.title("Disagreement Rate by Week")
    plt.xlabel("Week")
    plt.ylabel("Disagreement (mean)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = OUTDIR / "disagreement_rate_weekly.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def main():
    fbk = load_feedback()
    if fbk.empty:
        print("No feedback rows found. Submit feedback in the app first.")
        return

    # 1) Weekly disagreement
    plot_weekly_disagreement(fbk)

    # 2) Per-user throughput & disagreement
    by_user = (
        fbk.groupby("user_id")
        .agg(rows=("name", "count"), disagree_rate=("disagree", "mean"))
        .reset_index()
    )
    by_user = by_user.sort_values("rows", ascending=False)
    print("\nPer-user throughput & disagreement rate:")
    print(by_user.to_string(index=False))

    # 3) Disagreement by model version
    by_ver = (
        fbk.groupby("model_version")["disagree"]
        .mean()
        .reset_index()
        .sort_values("disagree", ascending=False)
    )
    print("\nDisagreement rate by model version:")
    print(by_ver.to_string(index=False))

    # 4) Weekly class counts (chosen_label)
    fbk_nonat = fbk.dropna(subset=["week"])
    if not fbk_nonat.empty:
        class_week = (
            fbk_nonat.groupby(["week", "chosen_label"]).size().unstack(fill_value=0)
        )
        print("\nWeekly class counts in feedback (chosen_label):")
        print(class_week.tail().to_string())
    else:
        print("\nNo rows with valid week to compute class counts.")

    # 5) Hot tokens in disagreements
    err = fbk[fbk["disagree"] == True]
    if not err.empty:
        tokens = (
            err["name"]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
            .str.split()
            .explode()
        )
        print("\nTop tokens in disagreements:")
        print(tokens.value_counts().head(20).to_string())
    else:
        print("\nNo disagreements found.")


if __name__ == "__main__":
    main()
