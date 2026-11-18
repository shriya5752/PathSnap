
from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    log_loss
)

from preprocessing.create_features import FEATURE_COLUMNS
from models.predict import score_candidates


TEST_PATH = Path("results/person1/student_test_labeled.csv")
OUTPUT_PATH = Path("results/person2")


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    columns = [
        "trip_id",
        "observation_id",
        "arc_id",
        "label",
        *FEATURE_COLUMNS
    ]

    print("Loading student-test data...", flush=True)

    test = pd.read_csv(
        TEST_PATH,
        usecols=columns,
        dtype={"trip_id": str}
    )

    if test["trip_id"].nunique() != 10:
        raise ValueError("Expected 10 student-test trips")

    test = test.dropna(subset=["label"]).copy()

    if not test["label"].isin([0, 1]).all():
        raise ValueError("Invalid labels")

    print("Test candidate rows:", len(test), flush=True)

    start = time.perf_counter()

    scored = score_candidates(test)

    inference_time = time.perf_counter() - start

    probabilities = scored["candidate_probability"].to_numpy()

    y_true = test["label"].astype(int).to_numpy()
    y_pred = (probabilities >= 0.5).astype(int)

    temp = test[
        ["trip_id", "observation_id", "arc_id", "label"]
    ].copy()

    temp["probability"] = probabilities

    grouped = temp.groupby(
        ["trip_id", "observation_id"]
    )

    positives = grouped["label"].sum()

    if (positives > 1).any():
        raise ValueError("Multiple correct candidates found")

    best_indices = grouped["probability"].idxmax()

    selected = temp.loc[best_indices].copy()

    selected["correct"] = selected["label"].eq(1)

    total_observations = len(selected)
    misses = int((positives == 0).sum())

    per_trip = selected.groupby("trip_id")["correct"].mean()

    results = {
        "model": "HistGradientBoosting",
        "training_type": "full_training_partition",
        "training_candidate_rows": 1429420,
        "test_candidate_rows": len(test),
        "evaluated_observations": total_observations,
        "top1_accuracy": float(selected["correct"].mean()),
        "macro_trip_top1": float(per_trip.mean()),
        "candidate_misses": misses,
        "candidate_miss_rate": misses / total_observations,
        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "precision": precision_score(
            y_true, y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred,
            zero_division=0
        ),
        "log_loss": log_loss(
            y_true, probabilities,
            labels=[0, 1]
        ),
        "inference_seconds": inference_time
    }

    output = pd.DataFrame([results])

    output.to_csv(
        OUTPUT_PATH / "full_model_student_test.csv",
        index=False
    )

    print("\nFULL-TRAINED MODEL EVALUATION")
    print(output.to_string(index=False))
    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    main()
