"""Compare independent candidate choices and Viterbi on student-test trips."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

from .evaluation import evaluate_prediction
from .inference import TripStore, predict_trip
from .scoring import GeometryScorer, ModelScorer


def compare(store, scorer):
    rows = []
    for trip_id in store.trip_ids():
        directory = store.directory(trip_id)
        start = perf_counter()
        prediction = predict_trip(directory, scorer)
        elapsed = perf_counter() - start
        metrics = evaluate_prediction(prediction, directory)
        baseline = metrics.pop("classifier_only")
        row = {"trip_id": trip_id, "model": scorer.name, "inference_seconds": round(elapsed, 3)}
        for key in ("observations", "eligible_pairs", "usable_observations", "candidate_misses", "candidate_miss_rate"):
            row[key] = metrics[key]
        for name, result in [("classifier", baseline), ("viterbi", metrics)]:
            for key in ("correct_observations", "pointwise_accuracy", "connected_pairs", "route_continuity", "sequence_similarity", "route_blocks"):
                row[f"{name}_{key}"] = result[key]
        rows.append(row)
        print(f"{trip_id}  {elapsed:.1f}s  {len(prediction.trip.track)} observations", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_test_dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.student_test_dir.resolve().name != "student_test":
        parser.error("Point the comparison at the supplied student_test folder.")
    scorer = ModelScorer(args.model) if args.model else GeometryScorer()
    rows = compare(TripStore(args.student_test_dir), scorer)
    if not rows:
        parser.error("No trips were found.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
