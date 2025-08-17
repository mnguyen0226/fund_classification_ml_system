# analyze_training.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TRAIN_LOG = ROOT / "models" / "train_log.csv"
OUT = ROOT / "analytics"
OUT.mkdir(exist_ok=True, parents=True)


def main():
    if not TRAIN_LOG.exists():
        print("No train_log.csv found. Run train_model.py at least once.")
        return
    df = pd.read_csv(TRAIN_LOG, parse_dates=["timestamp_utc"])
    if df.empty:
        print("Train log is empty.")
        return
    df = df.sort_values("timestamp_utc")

    plt.figure()
    plt.plot(df["timestamp_utc"], df["acc"], marker="o")
    plt.title("Model Accuracy Over Time")
    plt.xlabel("Retrain Timestamp (UTC)")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = OUT / "training_accuracy_over_time.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    print("\nLatest retrains:")
    print(
        df.tail(10)[
            [
                "timestamp_utc",
                "version",
                "model_file",
                "acc",
                "n_total",
                "n_train",
                "n_test",
            ]
        ]
    )


if __name__ == "__main__":
    main()
