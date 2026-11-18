# PathSnap Part 3

The API reads real GPS and road data, uses Person 1's candidate features, and applies Viterbi over directed road arcs. The original Review 1 demo is also available when a data folder is unset.

## Run locally

Run these commands in PowerShell from the repository root. Install dependencies if needed with `python -m pip install -r backend/requirements.txt`. Person 1's candidate shortlist uses SciPy when available. The bundled NumPy fallback calculates the same nearest neighbours with deterministic index tie-breaking, so exact ties may be resolved differently.

```powershell
# Real-trip demo. Replace the path with the extracted student_test folder.
$env:PATHSNAP_TRIP_ROOT = 'C:\path\to\Student folder\student_test'
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Unzip `Student folder.zip` and set `PATHSNAP_TRIP_ROOT` to its `Student folder/student_test` subdirectory. The committed [per-trip geometry baseline results](../results/geometry_baseline_student_test.csv) cover all 10 student-test trips. The trained team model is loaded when `PATHSNAP_MODEL` is set.

The Vite origin `http://localhost:5173` is allowed by default. Open [FastAPI docs](http://127.0.0.1:8000/docs) or run the following from another PowerShell window.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/trips
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body '{"trip_id":"00000000"}' |
  ConvertTo-Json -Depth 6
```

Unset `PATHSNAP_TRIP_ROOT` to serve only the synthetic Review 1 trip `00000001`. Run `python -m backend.demo` for the standalone toy comparison.

## API contract

| Request | Response |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `GET /trips` | A JSON array of available trip ID strings. |
| `POST /predict` | Accepts `{"trip_id":"00000000"}`. Returns `trip_id`, `gps_points`, `predicted_route`, `predicted_arc_ids`, `metrics` and `model_name`. Unknown IDs return 404. |

Coordinates are pairs in `[latitude, longitude]` order and road arcs retain their original integer IDs. When a trip has multiple disconnected blocks, `predicted_arc_ids` includes all blocks in chronological order. The current single-polyline contract displays the longest contiguous block by road distance, which avoids drawing across gaps. The response reports the decoded block count in `metrics.route_blocks`.

Synthetic mode returns `metrics: null`. Real mode always reports computed continuity and coverage. Set `PATHSNAP_EVALUATE_REFERENCE=1` only for permitted server-side evaluation to enable pointwise accuracy and sequence similarity when the trip has a reference. The frontend's `metrics.accuracy` field mirrors `metrics.pointwise_accuracy`. Reference files are read after prediction and their contents are never returned in the API. If no reference is present, reference-dependent metrics are null.

## Person 2 model handoff

Real mode defaults to an untrained geometry scorer using distance and heading. To use Person 2's trained HistGradientBoosting classifier, install the model dependencies and set the artifact path before starting FastAPI. The trained backend was verified with Python 3.12 and scikit-learn 1.9.1.

```powershell
python -m pip install -r backend/requirements-model.txt
$env:PATHSNAP_MODEL = (Resolve-Path 'artifacts/person2/hgb_full_train.joblib').Path
```

The backend calls `models.predict.score_candidates` with Person 1's eight features. It aligns the returned `candidate_probability` values on `trip_id`, `observation_id`, and `arc_id` before Viterbi uses them as emissions. The same scores supply the classifier-only comparison. The artifact is Person 2's fitted classifier with positive class `1` and scikit-learn version `1.9.1`.

## Evaluate the decoder

```powershell
python -m backend.evaluate $env:PATHSNAP_TRIP_ROOT --out results/geometry_baseline_student_test.csv
# Compare Person 2's classifier with and without decoding.
python -m backend.evaluate $env:PATHSNAP_TRIP_ROOT --model $env:PATHSNAP_MODEL --out results/hgb_student_test_viterbi.csv
```

Viterbi adds log candidate scores and a transition score based on GPS displacement versus shortest directed network distance. The search includes the partial lengths of the start and end arcs and any intermediate arcs. The defaults allow 15 m of same-arc backward GPS jitter, cap network travel at `55 m/s × elapsed time + 35 m`, and divide absolute distance disagreement by 30 m in the transition score. Gaps over 10 seconds, nonpositive times, missing candidates and unreachable states split the decoding block. These are initial engineering defaults.

Both methods use the same candidates and transitions. The independent baseline chooses the top candidate per point, then includes reachable intermediate arcs for route comparison. Continuity is the fraction of adjacent observation pairs with candidates on both ends and a time gap in `(0, 10]` seconds whose choices can be connected. Pointwise accuracy uses Person 1's usable reference alignments, with missed reference candidates counted as incorrect. Sequence similarity is one minus the arc-level edit distance divided by the longer sequence length. All sequence comparisons use the full ordered reference. These reference-aligned labels are weak labels.

The [trained-model comparison CSV](../results/hgb_student_test_viterbi.csv) covers all 10 student-test trips. Pointwise accuracy rose from 88.45% to 93.23%, continuity from 88.58% to 99.92%, and mean route sequence similarity from 42.39% to 73.29% with Viterbi.

## Two-laptop rehearsal

On Laptop 1, set the trip root and use `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000` on the agreed demo network. Add the exact browser origin serving the UI to `PATHSNAP_EXTRA_ORIGINS` if it is different from `http://localhost:5173`. On Laptop 2, set the frontend's API base URL to Laptop 1's LAN address and verify all three endpoints, the map, and the backend's request logs. For the final demo, point the backend at the organizer-provided evaluator trip directory and enable the permitted server-side evaluation flag.
