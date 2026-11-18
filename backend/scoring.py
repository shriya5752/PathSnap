"""Candidate emissions for development and for a fitted classifier."""

from pathlib import Path

import numpy as np


class GeometryScorer:
    name = "Geometry baseline + HMM/Viterbi"

    def scores(self, candidates):
        distance = candidates["distance_m"].to_numpy() / 20.0
        heading = candidates["heading_diff_deg"] * candidates["heading_valid"] / 75.0
        return np.exp(-distance - heading.to_numpy())


class ModelScorer:
    """Use Person 2's classifier probabilities as HMM emissions."""

    keys = ["trip_id", "observation_id", "arc_id"]

    def __init__(self, path):
        from models.predict import load_model, score_candidates

        self.path = str(Path(path).resolve())
        self.score_candidates = score_candidates
        model = load_model(self.path)
        self.name = f"{type(model).__name__} + HMM/Viterbi"

    def scores(self, candidates):
        scored = self.score_candidates(candidates, model_path=self.path)
        probabilities = candidates[self.keys].merge(
            scored, on=self.keys, how="left", validate="one_to_one",
        )["candidate_probability"]
        if len(scored) != len(candidates) or probabilities.isna().any():
            raise ValueError("Model probabilities must match every candidate ID exactly once.")
        return probabilities.to_numpy()
