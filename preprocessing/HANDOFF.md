# Person 1 handoff and design notes

## Data

`.track`: longitude, latitude, seconds; `.nodes`: longitude, latitude; `.arcs`: directed from/to zero-based node row IDs; `.route`: ordered zero-based arc row IDs. IDs are local to a trip. Trip IDs must be read as strings (`dtype={'trip_id': str}`). Observation IDs are zero-based track row numbers. Coordinates are checked for finiteness, range, and valid references. Original row order is retained; nonpositive time gaps are flagged, not silently reordered.

## Person 2: model training

Run the README command. `results/person1/train_labeled.csv` contains training candidates; `student_test_labeled.csv` is for evaluation only. Import `FEATURE_COLUMNS` from `preprocessing.create_features` and use only those columns as model inputs. Never use IDs, label status, reference IDs or route information as features. Filter training rows to usable labels, and prefer groups containing a positive candidate for conditional candidate classification. Candidate-miss groups must still count as failures in end-to-end Top-1 evaluation. Tune exclusively within training trips using group-aware CV. `*_alignment.csv` has one row per observation with reference arc and confidence status. Use it to calculate candidate miss rate among usable observations, and report usable fraction and ambiguous/far exclusions. Candidate classification and observation-level accuracy are not the same metric.

## Person 3: inference

```python
from preprocessing.load_data import load_trip
from preprocessing.create_features import candidate_features, FEATURE_COLUMNS

trip = load_trip(trip_directory)  # does NOT open .route
candidates = candidate_features(trip)  # k=10, shortlist=64
X = candidates[FEATURE_COLUMNS]
# model scores align with candidate rows; group by observation_id
# trip.nodes and trip.arcs preserve directed node and arc IDs
```

Do not invoke reference_alignment at prediction time. Use `load_reference` only in a separate evaluator after prediction. Candidate generation uses local equirectangular coordinates centered at mean GPS latitude, with Earth's mean radius. This converts degrees to approximate metres and is appropriate for local trips; projection error increases with geographic extent. Distances are point-to-segment, not point-to-midpoint. A KD-tree retrieves 64 nearby arc midpoints plus edges incident to eight nearby nodes, then ranks exact segment distances within that shortlist. The shortlist is approximate and can miss a long segment; candidate miss rate quantifies this risk together with label quality. A fixed top 10 is a starting configuration, not an optimized result.

Features: `distance_m` (point-to-segment), `heading_diff_deg` (directed motion vs arc, 0–180), `speed_mps` (previous observed displacement / seconds), `segment_length_m`, `time_gap_s`, `projection_fraction` (0–1 along directed arc), `heading_valid`, `time_valid`. First observation or invalid gap has zero speed/gap and indicator flags; zero displacement also invalidates heading. Missing values or invalid coordinates/IDs raise a descriptive error instead of silently corrupting IDs.

## Weak-label design

Reference route is a sequence, not an observation-sized label vector. A dynamic program aligns GPS observations to nondecreasing route occurrence indices by minimizing summed squared distances to reference segments. It permits staying on an arc or skipping route entries, preserves repeated IDs, and does not require the first/last GPS observation to correspond exactly to the first/last route arc. It is a weak spatial alignment, not independently verified truth. If selected segment is more than 50 m away the status is `far`; if another distinct reference arc is within 2 m of the selected distance (or closer) it is `ambiguous`; otherwise `usable`. These are explicit initial heuristic thresholds, not optimized. Ambiguous/far candidate labels are blank, not zero. If usable reference is missing from candidates, every candidate is negative and the miss is separately counted. Reference labels never influence candidate generation.

## Person 4: report materials

`dataset_summary.csv` contains trip-level counts, sizes, validation, timestamp gaps, extent and weak connectivity; `label_quality.csv` counts usability and candidate misses; `dataset_overview.png` plots distributions. Locally generated trajectory PNGs and Folium HTML maps show GPS, road network, and reference. `create_map(..., predicted=arc_ids)` adds predicted segments once Person 3 supplies them. Use genuine measured results and disclose AI assistance according to the assessment policy; inspect the code and rerun independently before submission.
