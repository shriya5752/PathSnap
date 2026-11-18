
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import HistGradientBoostingClassifier

from preprocessing.create_features import FEATURE_COLUMNS


DATA_PATH = Path("results/person1/train_labeled.csv")
OUTPUT_PATH = Path("artifacts/person2")
SEED = 42


def load_training_data():
    columns = [
        "trip_id",
        "observation_id",
        "label",
        *FEATURE_COLUMNS
    ]

    dtypes = {
        "trip_id": str,
        "observation_id": "int32",
        "label": "float32"
    }

    for feature in FEATURE_COLUMNS:
        dtypes[feature] = "float32"

    print("Loading training data...", flush=True)

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns,
        dtype=dtypes
    )

    if df["trip_id"].nunique() != 80:
        raise ValueError("Expected exactly 80 training trips")

    if not df["label"].dropna().isin([0, 1]).all():
        raise ValueError("Unexpected training labels")

    df = df.dropna(subset=["label"]).copy()

    positive_counts = df.groupby(
        ["trip_id", "observation_id"]
    )["label"].transform("sum")

    df = df[positive_counts == 1].copy()

    if not np.isfinite(
        df[FEATURE_COLUMNS].to_numpy()
    ).all():
        raise ValueError("Non-finite feature values found")

    x = df[FEATURE_COLUMNS]
    y = df["label"].astype("int8")

    observations = df.groupby(
        ["trip_id", "observation_id"]
    ).ngroups

    print("Training trips:", df["trip_id"].nunique())
    print("Training candidate rows:", len(df))
    print("Training GPS observations:", observations)
    print("Positive candidates:", int(y.sum()))
    print("Negative candidates:", int((y == 0).sum()))

    return x, y, observations


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    x, y, observations = load_training_data()

    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=31,
        random_state=SEED
    )

    print("\nStarting full-data training...", flush=True)

    start = time.perf_counter()

    model.fit(x, y)

    training_seconds = time.perf_counter() - start

    print(
        "Training completed in",
        round(training_seconds, 2),
        "seconds"
    )

    model_path = OUTPUT_PATH / "hgb_full_train.joblib"

    joblib.dump(model, model_path)

    metadata = {
        "model_name": "HistGradientBoostingClassifier",
        "model_status": "integration_candidate",
        "training_partition": "80_train_trips",
        "training_candidate_rows": len(x),
        "training_observations": observations,
        "feature_columns": FEATURE_COLUMNS,
        "python_model_library": "scikit-learn",
        "sklearn_version": sklearn.__version__,
        "random_state": SEED,
        "parameters": {
            "max_iter": 100,
            "max_leaf_nodes": 31
        },
        "training_seconds": training_seconds,
        "student_test_evaluated": False
    }

    metadata_path = OUTPUT_PATH / "model_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    loaded_model = joblib.load(model_path)

    sample = x.iloc[:10]
    probabilities = loaded_model.predict_proba(sample)

    if probabilities.shape != (10, 2):
        raise ValueError("Model verification failed")

    if not np.isfinite(probabilities).all():
        raise ValueError("Invalid prediction probabilities")

    print("\nMODEL EXPORT SUCCESSFUL")
    print("Model:", model_path)
    print("Metadata:", metadata_path)
    print("Verification predictions:", len(probabilities))


if __name__ == "__main__":
    main()
