# prep_retrain_dataset.py
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "data" / "seed" / "fund_nonfund.csv"
FDBK_DIR = ROOT / "feedback"
CURATED = ROOT / "data" / "curated"
CURATED.mkdir(parents=True, exist_ok=True)
OUT = CURATED / "training_dataset.csv"

VALID = {"Fund", "Non-Fund"}


def normalize_name(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def load_feedback() -> pd.DataFrame:
    frames = []
    for p in FDBK_DIR.rglob("labels_*.csv"):
        try:
            df = pd.read_csv(p)
            df["__file"] = str(p)
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Normalize column names (back-compat)
    if "predicted_label" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "predicted_label"})
    for col in ("name", "predicted_label", "correct_label", "timestamp_utc"):
        if col not in df.columns:
            df[col] = None

    # Final label per feedback row
    df["final_label"] = df.apply(
        lambda r: (
            r["correct_label"]
            if pd.notna(r.get("correct_label"))
            and str(r["correct_label"]).strip() != ""
            else r["predicted_label"]
        ),
        axis=1,
    )
    # Robust timestamps
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)

    # Fallback: derive from parent folder name YYYY-MM-DD if needed
    if ts.isna().any():

        def fallback(row):
            parts = Path(row["__file"]).parts
            for part in parts:
                if len(part) == 10 and part[4] == "-" and part[7] == "-":
                    return pd.to_datetime(part, errors="coerce", utc=True)
            return pd.NaT

        ts = ts.fillna(df.apply(fallback, axis=1))

    df["timestamp_utc"] = ts
    df["source"] = "feedback"
    df["name_norm"] = df["name"].apply(normalize_name)
    return df


def majority_vote(group: pd.DataFrame) -> pd.Series:
    """
    Majority vote over final_label; tie-break by most recent timestamp_utc.
    """
    sub = group.dropna(subset=["final_label"])
    sub = sub[sub["final_label"].isin(VALID)]
    if sub.empty:
        # keep last seen spelling even if invalid/missing
        return pd.Series(
            {
                "name": group.iloc[-1]["name"],
                "final_label": None,
                "timestamp_utc": pd.NaT,
                "source": "feedback",
            }
        )

    counts = Counter(sub["final_label"])
    top = counts.most_common()
    if len(top) == 1 or (len(top) > 1 and top[0][1] > top[1][1]):
        final_label = top[0][0]
        # timestamp: use the most recent among rows that chose the final label
        sub2 = sub[sub["final_label"] == final_label].copy()
    else:
        # tie → most recent row decides
        sub2 = sub.copy()

    sub2 = sub2.sort_values("timestamp_utc")
    ts = sub2.iloc[-1]["timestamp_utc"]
    name_str = group.iloc[-1]["name"]  # last-seen spelling
    return pd.Series(
        {
            "name": name_str,
            "final_label": sub2.iloc[-1]["final_label"],
            "timestamp_utc": ts,
            "source": "feedback",
        }
    )


def main():
    # ---------- Seed ----------
    seed = pd.read_csv(SEED)
    seed = seed.rename(columns={"label": "final_label"})
    seed["name_norm"] = seed["name"].apply(normalize_name)
    seed["source"] = "seed"

    # Seed timestamp: derive from file mtime so we don't need to change seed CSV schema
    try:
        mtime = datetime.fromtimestamp(
            SEED.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    except Exception:
        mtime = datetime.now(timezone.utc).isoformat()
    seed["timestamp_utc"] = mtime

    seed = seed[["name", "final_label", "source", "timestamp_utc", "name_norm"]]

    # ---------- Feedback ----------
    fbk = load_feedback()

    if not fbk.empty:
        # Majority vote per normalized name
        resolved = fbk.groupby("name_norm", as_index=False).apply(majority_vote)
        resolved = resolved.dropna(subset=["final_label"])
        resolved = resolved[resolved["final_label"].isin(VALID)]
        resolved = resolved[["name", "final_label", "source", "timestamp_utc"]]
        # map name_norm for joining
        resolved["name_norm"] = resolved["name"].apply(normalize_name)
    else:
        resolved = pd.DataFrame(
            columns=["name", "final_label", "source", "timestamp_utc", "name_norm"]
        )

    # ---------- Merge (feedback overrides seed by name_norm) ----------
    seed_keep = ~seed["name_norm"].isin(set(resolved["name_norm"]))
    curated = pd.concat(
        [
            seed.loc[seed_keep, ["name", "final_label", "source", "timestamp_utc"]],
            resolved[["name", "final_label", "source", "timestamp_utc"]],
        ],
        ignore_index=True,
    )

    # Sanity filter
    curated = curated[curated["final_label"].isin(VALID)]

    # Write curated snapshot
    curated.to_csv(OUT, index=False)
    print(f"Wrote curated dataset → {OUT} (rows={len(curated)})")
    print("Columns:", list(curated.columns))


if __name__ == "__main__":
    main()
