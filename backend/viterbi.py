"""Decode the most likely road sequence for an observation sequence."""

from collections.abc import Hashable, Mapping, Sequence
from math import log
from typing import TypeVar

ArcId = TypeVar("ArcId", bound=Hashable)


def decode(
    emissions: Sequence[Mapping[ArcId, float]],
    transitions: Mapping[ArcId, Mapping[ArcId, float]],
) -> list[ArcId]:
    """Use a uniform initial prior and treat absent transitions as impossible."""
    if not emissions:
        return []

    scores = {arc: log(score) for arc, score in emissions[0].items() if score > 0}
    if not scores:
        raise ValueError("No candidate has a positive score at observation 1.")

    predecessors = []
    for observation, candidates in enumerate(emissions[1:], start=2):
        best_scores = {}
        best_previous = {}
        for previous_arc, previous_score in scores.items():
            for arc, probability in transitions.get(previous_arc, {}).items():
                emission = candidates.get(arc, 0)
                if probability <= 0 or emission <= 0:
                    continue
                score = previous_score + log(probability) + log(emission)
                if arc not in best_scores or score > best_scores[arc]:
                    best_scores[arc] = score
                    best_previous[arc] = previous_arc

        if not best_scores:
            raise ValueError(f"No valid route reaches observation {observation}.")
        scores = best_scores
        predecessors.append(best_previous)

    path = [max(scores, key=scores.__getitem__)]
    for previous in reversed(predecessors):
        path.append(previous[path[-1]])
    return path[::-1]
