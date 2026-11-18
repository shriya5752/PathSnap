"""Candidate emissions for development and for a fitted classifier."""

import numpy as np

from preprocessing.create_features import FEATURE_COLUMNS


class GeometryScorer:
    name = "Geometry baseline + HMM/Viterbi"

    def scores(self, candidates):
        distance = candidates["distance_m"].to_numpy() / 20.0
        heading = candidates["heading_diff_deg"] * candidates["heading_valid"] / 75.0
        return np.exp(-distance - heading.to_numpy())


class ModelScorer:
    """Load a trusted Joblib dict with model, feature_columns and positive_class."""

    def __init__(self, path):
        import joblib

        artifact = joblib.load(path)
        model = artifact["model"]
        columns = list(artifact["feature_columns"])
        if len(columns) != len(FEATURE_COLUMNS) or set(columns) != set(FEATURE_COLUMNS):
            raise ValueError("Model feature columns must match Person 1's feature schema.")
        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError("The fitted pipeline must implement predict_proba.")
        self.positive_index = list(model.classes_).index(artifact["positive_class"])
        self.model = model
        self.columns = columns
        self.name = f"{artifact.get('model_name') or type(model).__name__} + HMM/Viterbi"

    def scores(self, candidates):
        return self.model.predict_proba(candidates[self.columns])[:, self.positive_index]
