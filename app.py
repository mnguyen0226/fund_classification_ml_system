# app.py
from pathlib import Path
import uuid, re
from datetime import datetime, timezone
import pandas as pd
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
ACTIVE_TXT = MODELS / "active_model.txt"
FEEDBACK_DIR = ROOT / "feedback"
FEEDBACK_DIR.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="Fund Name Classifier", page_icon="🏦", layout="centered")

# ---- Reset handling (must run BEFORE widgets are created) ----
# If a previous run requested a reset, clear widget/state keys here *before* instantiating widgets
if st.session_state.get("__reset", False):
    for k in ("editable_df", "editable_editor", "names_text"):
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["__reset"] = False

# ---- Light styling ----
st.markdown(
    """
<style>
.app-card { background: white; border-radius: 16px; padding: 20px 22px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.07); border: 1px solid rgba(0,0,0,0.06);}
.app-title { font-size: 2rem !important; font-weight: 700 !important; margin-bottom: .25rem !important; }
.app-subtitle { color: #6b7280; margin-bottom: 1.0rem; }
.stDataFrame [data-testid="stTable"] thead tr th:first-child {display:none}
.stDataFrame [data-testid="stTable"] tbody tr td:first-child {display:none}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-title">🏦 Fund vs Non-Fund Classifier</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Paste names (one per line), review predictions, and correct labels inline.</div>',
    unsafe_allow_html=True,
)


# ---- Model helpers ----
def _latest_model_path():
    files = sorted(MODELS.glob("fund_classifier_*.pkl"))
    return files[-1] if files else None


def _active_model_path():
    if ACTIVE_TXT.exists():
        name = ACTIVE_TXT.read_text().strip()
        p = MODELS / name
        if p.exists():
            return p
    return _latest_model_path()


@st.cache_resource
def load_model():
    p = _active_model_path()
    if not p:
        raise FileNotFoundError("No model found. Train a model first.")
    model = joblib.load(p)
    return model, p.name


def safe_for_filename(s: str) -> str:
    s = (s or "unknown").strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:60]


# ---- Username ----
user_id = st.text_input("Your username", placeholder="e.g., minh.nguyen").strip()
if not user_id:
    st.info("Enter your username to enable prediction & feedback.")

# ---- Load model once ----
try:
    model, model_file = load_model()
    st.caption(f"Active model: `{model_file}`")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------- Card ----------
with st.container():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)

    # Use a key so we can clear it via the reset flag on the next run
    text = st.text_area(
        "Names (one per line)",
        key="names_text",
        value=st.session_state.get("names_text", ""),
        height=220,
        placeholder="e.g.\nArcadia Global Equity Fund\nAcme Inc.\nOlivia Nguyen",
        label_visibility="visible",
    )

    c1, c2 = st.columns(2)
    with c1:
        run_btn = st.button(
            "Predict", use_container_width=True, disabled=(not bool(user_id))
        )
    with c2:
        if st.button("Clear", use_container_width=True):
            st.session_state["__reset"] = True
            st.rerun()

    # On predict: build ONE editable dataframe: name, predicted_label (locked), correct_label (editable), comments (editable)
    if run_btn and user_id:
        rows = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not rows:
            st.warning("Please paste at least one non-empty line.")
        else:
            preds = model.predict(rows)
            editable = pd.DataFrame(
                {
                    "name": rows,
                    "predicted_label": preds,
                    "correct_label": preds,  # default to predicted
                    "comments": [""] * len(rows),
                }
            )
            st.session_state["editable_df"] = editable

    # ----- Single editable table & submit -----
    if "editable_df" in st.session_state:
        st.markdown("**Review & correct labels**")
        with st.form("feedback_form", clear_on_submit=False):
            edited = st.data_editor(
                st.session_state["editable_df"],
                key="editable_editor",
                use_container_width=True,
                num_rows="fixed",
                column_order=["name", "predicted_label", "correct_label", "comments"],
                column_config={
                    "name": st.column_config.TextColumn("Name", disabled=True),
                    "predicted_label": st.column_config.TextColumn(
                        "Predicted", disabled=True
                    ),
                    "correct_label": st.column_config.SelectboxColumn(
                        "Correct Label", options=["Fund", "Non-Fund"]
                    ),
                    "comments": st.column_config.TextColumn("Comments"),
                },
                disabled=["name", "predicted_label"],
            )
            submitted = st.form_submit_button(
                "Submit feedback", type="primary", use_container_width=True
            )

        if submitted:
            final_fb = edited.copy()
            ts = datetime.now(timezone.utc)
            date_dir = FEEDBACK_DIR / ts.strftime("%Y-%m-%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            batch_id = str(uuid.uuid4())

            fname = f"labels_{ts.strftime('%Y%m%dT%H%M%SZ')}_user={safe_for_filename(user_id)}_batch={batch_id[:8]}.csv"
            path = date_dir / fname

            final_fb = final_fb.assign(
                timestamp_utc=ts.isoformat(),
                batch_id=batch_id,
                model_version=model_file.replace("fund_classifier_", "").replace(
                    ".pkl", ""
                ),
                source="paste_batch",
                user_id=user_id,
                client_version="app_v1",
                row_id=[str(uuid.uuid4()) for _ in range(len(final_fb))],
            )
            final_fb.to_csv(path, index=False, encoding="utf-8")
            st.success(f"Thanks! Saved feedback ({len(final_fb)} rows).")

            # Request a full reset on the next run (so we clear textarea + table safely)
            st.session_state["__reset"] = True
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("⚠️ Mock demo with synthetic data — not for real compliance decisions.")
