
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
OBSERVATIONS_PER_TRIP = 60
N_SPLITS = 5


def load_data():
    columns = [
        "trip_id", "observation_id", "arc_id",
        "label", *FEATURE_COLUMNS
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns,
        dtype={"trip_id": str}
    )

    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Non-finite feature values found")

    if not df["label"].dropna().isin([0, 1]).all():
        raise ValueError("Invalid labels found")

    df = df.dropna(subset=["label"]).copy()

    grouped = df.groupby(
        ["trip_id", "observation_id"]
    )["label"].sum()

    valid = grouped[grouped == 1].index

    df = df.set_index(["trip_id", "observation_id"])
    df = df.loc[df.index.isin(valid)].reset_index()

    selected = []

    for trip_id, trip in df.groupby("trip_id"):
        ids = trip["observation_id"].drop_duplicates()

        sampled_ids = ids.sample(
            n=min(OBSERVATIONS_PER_TRIP, len(ids)),
            random_state=SEED
        )

        selected.append(
            trip[trip["observation_id"].isin(sampled_ids)]
        )

    data = pd.concat(selected, ignore_index=True)

    print("Trips:", data["trip_id"].nunique())
    print("Candidate rows:", len(data))
    print(
        "Observations:",
        data.groupby(["trip_id", "observation_id"]).ngroups
    )

    return data


def get_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=SEED
            )
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
            n_jobs=2,
            random_state=SEED
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=31,
            random_state=SEED
        )
    }


def top1_accuracy(data, probabilities):
    temp = data[
        ["trip_id", "observation_id", "arc_id", "label"]
    ].copy()

    temp["probability"] = probabilities

    best = temp.loc[
        temp.groupby(
            ["trip_id", "observation_id"]
        )["probability"].idxmax()
    ]

    return float((best["label"] == 1).mean())


def evaluate(model, train, validation):
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
            y_valid, predictions,
            average="macro", zero_division=0
        ),
        "precision": precision_score(
            y_valid, predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_valid, predictions,
            zero_division=0
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

    data = load_data()
    models = get_models()

    if data["trip_id"].nunique() != 80:
        raise ValueError("Expected 80 training trips")

    x = data[FEATURE_COLUMNS]
    y = data["label"].astype(int)
    groups = data["trip_id"]

    cv = GroupKFold(n_splits=N_SPLITS)
    splits = list(cv.split(x, y, groups))

    results = []

    for name, model in models.items():
        print(f"\nMODEL: {name}", flush=True)

        for fold, (train_idx, valid_idx) in enumerate(
            splits, start=1
        ):
            train = data.iloc[train_idx]
            validation = data.iloc[valid_idx]

            train_trips = set(train["trip_id"])
            valid_trips = set(validation["trip_id"])

            if not train_trips.isdisjoint(valid_trips):
                raise ValueError("Trip leakage detected")

            metrics = evaluate(model, train, validation)

            results.append({
                "model": name,
                "fold": fold,
                "train_rows": len(train),
                "validation_rows": len(validation),
                **metrics
            })

            print(
                f"Fold {fold}/{N_SPLITS} | "
                f"Macro F1: {metrics['macro_f1']:.4f} | "
                f"Top-1: {metrics['top1_accuracy']:.4f}",
                flush=True
            )

            pd.DataFrame(results).to_csv(
                OUTPUT_PATH / "sampled_80trip_cv_results.csv",
                index=False
            )

    results_df = pd.DataFrame(results)

    columns = [
        "macro_f1", "precision", "recall",
        "log_loss", "top1_accuracy",
        "training_seconds", "inference_seconds"
    ]

    summary = results_df.groupby("model")[columns].agg(
        ["mean", "std"]
    )

    summary.to_csv(
        OUTPUT_PATH / "sampled_80trip_summary.csv"
    )

    print("\nEXPERIMENT COMPLETE")
    print(summary.round(4))


if __name__ == "__main__":
    main()
