# PathSnap Part 3 Handoff

The trained HistGradientBoosting classifier scores Person 1's eight features for each candidate road. The backend matches probabilities by trip ID, observation ID, and arc ID. Viterbi uses these probabilities with directed road-network transitions scored by network distance, GPS displacement, and elapsed time. Missing candidates, long gaps, and impossible moves split the route. The current map response shows the longest connected section and returns the original arc IDs from all sections.

## Student-test comparison

Both methods used the same classifier and candidates on all ten student-test trips. Pointwise accuracy used 14,043 usable reference alignments. Route continuity measured the share of eligible adjacent observation pairs joined by a feasible directed move. Mean sequence similarity used normalized arc edit distance averaged across trips.

| Metric | Classifier only | Classifier plus Viterbi |
|---|---:|---:|
| Pointwise accuracy | 88.45% | 93.23% |
| Route continuity | 88.58% | 99.92% |
| Mean route sequence similarity | 42.39% | 73.29% |

Continuity and sequence similarity improved on every trip. Pointwise accuracy fell by 3.70 percentage points on `00000007` and 5.18 percentage points on `00000065`. The comparison uses student-test data. The reserved evaluator trips require a separate final check when issued.

Reference `.route` files are kept out of prediction. The server opens them after prediction when reference evaluation is enabled. Trip `00000000` produced identical route output with evaluation enabled and disabled.

## Demo evidence

Laptop 1 made live HTTP requests to its Wi-Fi address. `GET /health`, `GET /trips`, the Vite CORS preflight for `http://localhost:5173`, and all ten `POST /predict` requests returned HTTP 200. Trip `00000000` returned 2,503 GPS points, 262 predicted arcs, 97.78% pointwise accuracy, and 99.92% route continuity.

The backend receives a request from Laptop 2 during the final rehearsal. Use trip `00000000` and capture the request in the backend console and the reconstructed route in the frontend. The Laptop 1 checks establish server operation. Laptop 2 reachability remains to be verified.

[Trained-model results CSV](https://github.com/shriya5752/PathSnap/blob/main/results/hgb_student_test_viterbi.csv) · [Two-laptop setup](https://github.com/shriya5752/PathSnap/blob/main/frontend/DEPLOYMENT.md)
