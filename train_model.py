# train_model.py
from pathlib import Path
import argparse, json, hashlib
from datetime import datetime, timezone
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "curated" / "training_dataset.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True, parents=True)
ACTIVE_TXT = MODELS / "active_model.txt"
TRAIN_LOG = MODELS / "train_log.csv"


def stable_split_mask(names: pd.Series, test_frac: float = 0.2) -> pd.Series:
    thr = int(test_frac * 100)
    vals = (
        names.astype(str)
        .str.lower()
        .str.strip()
        .apply(lambda s: int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % 100)
    )
    return vals < thr


def main(promote: bool = False):
    df = pd.read_csv(DATA)
    if df.empty:
        raise SystemExit("Curated dataset is empty. Run prep_retrain_dataset.py first.")

    X = df["name"].astype(str)
    y = df["final_label"].astype(str)

    test_mask = stable_split_mask(X, test_frac=0.2)
    X_train, y_train = X[~test_mask], y[~test_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_df=0.95, lowercase=True
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    labels = ["Fund", "Non-Fund"]
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    ts = datetime.now(timezone.utc)
    version = ts.strftime("%Y%m%d_%H%M%S")
    model_file = MODELS / f"fund_classifier_{version}.pkl"
    joblib.dump(pipe, model_file)

    # Append to train_log.csv
    row = {
        "timestamp_utc": ts.isoformat(),
        "version": version,
        "model_file": model_file.name,
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "acc": round(acc, 6),
        "precision_Fund": round(
            report.get("Fund", {}).get("precision", float("nan")), 6
        ),
        "recall_Fund": round(report.get("Fund", {}).get("recall", float("nan")), 6),
        "precision_NonFund": round(
            report.get("Non-Fund", {}).get("precision", float("nan")), 6
        ),
        "recall_NonFund": round(
            report.get("Non-Fund", {}).get("recall", float("nan")), 6
        ),
    }
    if TRAIN_LOG.exists():
        pd.concat(
            [pd.read_csv(TRAIN_LOG), pd.DataFrame([row])], ignore_index=True
        ).to_csv(TRAIN_LOG, index=False)
    else:
        pd.DataFrame([row]).to_csv(TRAIN_LOG, index=False)

    # Optional Excel for convenience
    try:
        pd.read_csv(TRAIN_LOG).to_excel(MODELS / "train_log.xlsx", index=False)
    except Exception:
        pass

    print(f"Saved {model_file.name} | test acc={acc:.3f}")

    if promote:
        ACTIVE_TXT.write_text(model_file.name)
        print(f"Promoted active model -> {model_file.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--promote",
        action="store_true",
        help="Set this newly trained model as the active model",
    )
    args = ap.parse_args()
    main(promote=args.promote)
