"""Predict routes from GPS and the road network without accessing references."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from preprocessing.create_features import candidate_features, project
from preprocessing.load_data import Trip, load_trip

from .hmm import Comparison, decode_trip
from .routing import Candidate, DecoderConfig, RoadGraph


@dataclass
class Prediction:
    trip: Trip
    comparison: Comparison
    candidate_arc_ids: list[set[int]]
    arc_lengths: np.ndarray
    model_name: str

    def response(self, metrics):
        blocks = self.comparison.decoded.route_blocks
        route = []
        if blocks:
            # The agreed API has one polyline. Return the longest contiguous block.
            primary = max(blocks, key=lambda block: self.arc_lengths[block].sum())
            starts = self.trip.arcs[primary, 0]
            final = self.trip.arcs[primary[-1], 1]
            route = self.trip.nodes[np.append(starts, final)][:, ::-1].tolist()
        return {
            "trip_id": self.trip.trip_id,
            "gps_points": self.trip.track[:, :2][:, ::-1].tolist(),
            "predicted_route": route,
            "predicted_arc_ids": [arc for block in blocks for arc in block],
            "metrics": metrics,
            "model_name": self.model_name,
        }


class TripStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Trip folder does not exist: {self.root}")

    def trip_ids(self):
        return sorted(
            entry.name for entry in self.root.iterdir()
            if entry.is_dir() and re.fullmatch(r"[0-9]{8}", entry.name)
            and entry.resolve().parent == self.root
            and all((entry / f"{entry.name}.{suffix}").is_file()
                    for suffix in ("track", "nodes", "arcs"))
        )

    def directory(self, trip_id):
        if trip_id not in self.trip_ids():
            raise KeyError(trip_id)
        return self.root / trip_id


def predict_trip(directory, scorer, config=DecoderConfig()):
    trip = load_trip(directory)
    points, nodes = project(trip)
    lengths = np.linalg.norm(nodes[trip.arcs[:, 1]] - nodes[trip.arcs[:, 0]], axis=1)
    graph = RoadGraph(trip.arcs, lengths, len(nodes))
    groups = [[] for _ in trip.track]
    if len(trip.arcs):
        features = candidate_features(trip)
        if len(features):
            scores = np.asarray(scorer.scores(features), dtype=float)
            if scores.shape != (len(features),) or not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1)):
                raise ValueError("The scorer must return one finite probability per candidate row.")
            columns = features[["observation_id", "arc_id", "projection_fraction"]]
            for (observation, arc, fraction), score in zip(columns.itertuples(index=False, name=None), scores):
                groups[observation].append(Candidate(int(arc), float(fraction), float(score)))
    result = decode_trip(groups, graph, points, trip.track[:, 2], config)
    return Prediction(trip, result, [{c.arc_id for c in group} for group in groups], lengths, scorer.name)
