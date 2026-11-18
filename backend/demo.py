"""A synthetic trip with a tempting candidate on a disconnected road."""

from .viterbi import decode

TRIP_ID = "00000001"
GPS_POINTS = (
    (12.97008, 77.59016),
    (12.97032, 77.59043),
    (12.97036, 77.59077),
)
ARC_ENDPOINTS = {
    104: ((12.97010, 77.59010), (12.97020, 77.59030)),
    205: ((12.97020, 77.59030), (12.97030, 77.59060)),
    310: ((12.97030, 77.59060), (12.97040, 77.59085)),
    420: ((12.97032, 77.59033), (12.97033, 77.59054)),
}
ARC_LABELS = {104: "A", 205: "B", 310: "C", 420: "X"}
CANDIDATE_SCORES = (
    {104: 1.0},
    {205: 0.35, 420: 0.65},
    {310: 1.0},
)
TRANSITION_PROBABILITIES = {
    104: {205: 1.0},
    205: {310: 1.0},
}


def demo_arc_ids() -> list[int]:
    return decode(CANDIDATE_SCORES, TRANSITION_PROBABILITIES)


def run_demo() -> dict:
    independent = [
        max(candidates, key=candidates.__getitem__) for candidates in CANDIDATE_SCORES
    ]
    return {
        "trip_id": TRIP_ID,
        "mode": "synthetic_demo",
        "classifier_only_arc_ids": independent,
        "predicted_arc_ids": demo_arc_ids(),
    }


if __name__ == "__main__":
    result = run_demo()
    print("Middle observation scores: B=0.35, X=0.65")
    print("Allowed transitions: A -> B, B -> C. X is disconnected.")
    print("Classifier only:", " -> ".join(ARC_LABELS[arc] for arc in result["classifier_only_arc_ids"]))
    print("Viterbi:        ", " -> ".join(ARC_LABELS[arc] for arc in result["predicted_arc_ids"]))
