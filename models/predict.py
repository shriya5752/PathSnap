
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from preprocessing.create_features import FEATURE_COLUMNS


DEFAULT_MODEL_PATH = Path("artifacts/person2/hgb_full_train.joblib")


@lru_cache(maxsize=2)
def load_model(model_path):
    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support probabilities")

    if 1 not in model.classes_:
        raise ValueError("Model has no positive class")

    return model


def score_candidates(candidates, model_path=DEFAULT_MODEL_PATH):
    required = [
        "trip_id",
        "observation_id",
        "arc_id",
        *FEATURE_COLUMNS
    ]

    missing = [
        column for column in required
        if column not in candidates.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output_columns = [
        "trip_id",
        "observation_id",
        "arc_id",
        "candidate_probability"
    ]

    if candidates.empty:
        return pd.DataFrame(columns=output_columns)

    x = candidates[FEATURE_COLUMNS]

    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Candidate features contain invalid values")

    model = load_model(str(Path(model_path).resolve()))

    positive_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(x)[:, positive_index]

    output = candidates[
        ["trip_id", "observation_id", "arc_id"]
    ].copy()

    output["candidate_probability"] = probabilities

    return output
