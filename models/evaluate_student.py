
from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

from preprocessing.create_features import FEATURE_COLUMNS
from models.train_cv import load_data, get_models


TEST_PATH = Path("results/person1/student_test_labeled.csv")
OUTPUT_PATH = Path("results/person2")


def load_student_test():
    columns = [
        "trip_id", "observation_id", "arc_id",
        "label", *FEATURE_COLUMNS
    ]

    df = pd.read_csv(
        TEST_PATH,
        usecols=columns,
        dtype={"trip_id": str}
    )

    if df["trip_id"].nunique() != 10:
        raise ValueError("Expected 10 student-test trips")

    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Invalid feature values")

    if not df["label"].dropna().isin([0, 1]).all():
        raise ValueError("Invalid label values")

    df = df.dropna(subset=["label"]).copy()

    return df


def observation_results(df, probabilities):
    temp = df[
        ["trip_id", "observation_id", "arc_id", "label"]
    ].copy()

    temp["probability"] = probabilities

    grouped = temp.groupby(
        ["trip_id", "observation_id"]
    )

    positive_counts = grouped["label"].sum()

    if (positive_counts > 1).any():
        raise ValueError("Multiple positive candidates detected")

    best_indices = grouped["probability"].idxmax()
    selected = temp.loc[best_indices].copy()

    selected["correct"] = selected["label"].eq(1)

    total = len(selected)
    misses = int((positive_counts == 0).sum())

    per_trip = selected.groupby("trip_id")["correct"].mean()

    return {
        "top1_accuracy": float(selected["correct"].mean()),
        "macro_trip_top1": float(per_trip.mean()),
        "candidate_misses": misses,
        "evaluated_observations": total,
        "candidate_miss_rate": misses / total
    }


def evaluate(name, model, train, test):
    print(f"\nTraining {name}", flush=True)

    start = time.perf_counter()

    model.fit(
        train[FEATURE_COLUMNS],
        train["label"].astype(int)
    )

    training_time = time.perf_counter() - start

    print("Training complete", flush=True)

    start = time.perf_counter()

    probabilities = model.predict_proba(
        test[FEATURE_COLUMNS]
    )[:, 1]

    inference_time = time.perf_counter() - start

    y_true = test["label"].astype(int)
    y_pred = (probabilities >= 0.5).astype(int)

    metrics = observation_results(test, probabilities)

    metrics.update({
        "model": name,
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
        "training_seconds": training_time,
        "inference_seconds": inference_time
    })

    print(
        f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f}"
    )
    print(
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )
    print(
        f"Candidate Miss Rate: {metrics['candidate_miss_rate']:.4f}"
    )

    return metrics


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    train = load_data()
    test = load_student_test()

    print("Training trips:", train["trip_id"].nunique())
    print("Student-test trips:", test["trip_id"].nunique())
    print("Training candidate rows:", len(train))
    print("Test candidate rows:", len(test))

    models = get_models()

    selected_models = [
        "Random Forest",
        "HistGradientBoosting"
    ]

    results = []

    for name in selected_models:
        metrics = evaluate(
            name, models[name], train, test
        )

        results.append(metrics)

        pd.DataFrame(results).to_csv(
            OUTPUT_PATH / "student_test_leading_models.csv",
            index=False
        )

    print("\nSTUDENT-TEST EVALUATION COMPLETE")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
