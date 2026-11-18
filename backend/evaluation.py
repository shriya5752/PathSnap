"""Reference-based scoring, called only after inference."""

import numpy as np

from preprocessing.create_labels import reference_alignment
from preprocessing.load_data import load_reference


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _sequence_similarity(left, right):
    """One minus arc-level Levenshtein distance divided by the longer length."""
    prior = list(range(len(right) + 1))
    for i, arc in enumerate(left, start=1):
        current = [i]
        for j, reference_arc in enumerate(right, start=1):
            current.append(min(current[-1] + 1, prior[j] + 1, prior[j - 1] + (arc != reference_arc)))
        prior = current
    length = max(len(left), len(right))
    return 1 - prior[-1] / length if length else None


def _decoder_metrics(result, eligible, labels, reference):
    correct = sum(result.point_arcs[index] == arc for index, arc in labels)
    route = [arc for block in result.route_blocks for arc in block]
    return {
        "matched_observations": sum(arc is not None for arc in result.point_arcs),
        "route_blocks": len(result.route_blocks),
        "connected_pairs": result.valid_links,
        "route_continuity": _ratio(result.valid_links, eligible),
        "correct_observations": correct if reference is not None else None,
        "pointwise_accuracy": _ratio(correct, len(labels)),
        "sequence_similarity": _sequence_similarity(route, reference) if reference is not None else None,
    }


def evaluate_prediction(prediction, directory=None):
    reference = None
    labels = []
    if directory is not None and (directory / f"{prediction.trip.trip_id}.route").is_file():
        reference = load_reference(directory, len(prediction.trip.arcs)).tolist()
        alignment = reference_alignment(prediction.trip, np.asarray(reference))
        usable = alignment.loc[alignment.label_status == "usable"]
        labels = list(usable[["observation_id", "reference_arc_id"]].itertuples(index=False, name=None))

    result = prediction.comparison
    metrics = _decoder_metrics(result.decoded, result.eligible_links, labels, reference)
    misses = sum(arc not in prediction.candidate_arc_ids[index] for index, arc in labels)
    metrics.update({
        "accuracy": metrics["pointwise_accuracy"],
        "observations": len(prediction.trip.track),
        "eligible_pairs": result.eligible_links,
        "usable_observations": len(labels) if reference is not None else None,
        "candidate_misses": misses if reference is not None else None,
        "candidate_miss_rate": _ratio(misses, len(labels)),
        "classifier_only": _decoder_metrics(result.classifier, result.eligible_links, labels, reference),
    })
    return metrics
