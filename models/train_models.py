
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

from preprocessing.create_features import FEATURE_COLUMNS


DATA_PATH = Path("results/person1/train_labeled.csv")
OUTPUT_PATH = Path("results/person2")
SEED = 42
MAX_TRIPS = 12
MAX_OBSERVATIONS_PER_TRIP = 200


def load_pilot_data():
    df = pd.read_csv(DATA_PATH, dtype={"trip_id": str})

    required = [
        "trip_id", "observation_id", "arc_id",
        "label", *FEATURE_COLUMNS
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["label"]).copy()

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Unexpected label values")

    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Non-finite input features found")

    groups = df.groupby(["trip_id", "observation_id"])["label"].agg(
        ["sum", "count"]
    )

    valid = groups[(groups["sum"] == 1) & (groups["count"] > 0)]

    df = df.set_index(["trip_id", "observation_id"])
    df = df.loc[df.index.isin(valid.index)].reset_index()

    trips = sorted(df["trip_id"].unique())[:MAX_TRIPS]
    df = df[df["trip_id"].isin(trips)].copy()

    selected = []

    for trip_id, group in df.groupby("trip_id"):
        observation_ids = group["observation_id"].drop_duplicates()
        sampled = observation_ids.sample(
            n=min(MAX_OBSERVATIONS_PER_TRIP, len(observation_ids)),
            random_state=SEED
        )
        selected.append(group[group["observation_id"].isin(sampled)])

    df = pd.concat(selected, ignore_index=True)

    print("Pilot candidate rows:", len(df))
    print("Pilot observations:", df.groupby(
        ["trip_id", "observation_id"]
    ).ngroups)
    print("Pilot trips:", df["trip_id"].nunique())

    return df


def create_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7)
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=10,
            random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=2
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=31,
            random_state=SEED
        )
    }


def top1_accuracy(df, probabilities):
    predicted = df[["trip_id", "observation_id", "arc_id", "label"]].copy()
    predicted["probability"] = probabilities

    best = predicted.loc[
        predicted.groupby(["trip_id", "observation_id"])[
            "probability"
        ].idxmax()
    ]

    return float((best["label"] == 1).mean())


def evaluate_model(model, train, validation):
    x_train = train[FEATURE_COLUMNS]
    y_train = train["label"].astype(int)

    x_valid = validation[FEATURE_COLUMNS]
    y_valid = validation["label"].astype(int)

    start = time.perf_counter()
    model.fit(x_train, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    probabilities = model.predict_proba(x_valid)[:, 1]
    inference_time = time.perf_counter() - start

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "macro_f1": f1_score(
            y_valid, predictions, average="macro",
            zero_division=0
        ),
        "precision": precision_score(
            y_valid, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_valid, predictions, zero_division=0
        ),
        "log_loss": log_loss(
            y_valid, probabilities, labels=[0, 1]
        ),
        "top1_accuracy": top1_accuracy(
            validation, probabilities
        ),
        "training_seconds": training_time,
        "inference_seconds": inference_time
    }


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    df = load_pilot_data()
    models = create_models()

    splitter = GroupKFold(n_splits=3)
    x = df[FEATURE_COLUMNS]
    y = df["label"].astype(int)
    groups = df["trip_id"]

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}")

        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(x, y, groups), start=1
        ):
            train = df.iloc[train_idx]
            validation = df.iloc[valid_idx]

            assert set(train["trip_id"]).isdisjoint(
                set(validation["trip_id"])
            )

            metrics = evaluate_model(model, train, validation)

            results.append({
                "model": name,
                "fold": fold,
                **metrics
            })

            print(
                f"Fold {fold}: "
                f"F1={metrics['macro_f1']:.4f}, "
                f"Top-1={metrics['top1_accuracy']:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH / "pilot_cv_results.csv",
        index=False
    )

    summary = results_df.groupby("model")[
        ["macro_f1", "precision", "recall",
         "log_loss", "top1_accuracy",
         "training_seconds", "inference_seconds"]
    ].mean()

    summary.to_csv(
        OUTPUT_PATH / "pilot_summary.csv"
    )

    print("\nPILOT RESULTS")
    print(summary.round(4))
    print("\nSaved results in results/person2/")


if __name__ == "__main__":
    main()
