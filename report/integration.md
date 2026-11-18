# PathSnap trained integration

Tested on 18 November 2026 at baseline `ab0d9c1` on `feature/frontend-integration`. Cached `origin/main` matched HEAD; GitHub reported `fb2cd906e2c268544d0ad670294ecb77cc3579b3`. Remote ancestry must be checked after a manual fetch before declaring the branch current.

## Reproduce

From the repository root in PowerShell with a compatible, activated Python environment:

```powershell
python -m pip install -r backend/requirements-model.txt
$env:PATHSNAP_TRIP_ROOT = "$HOME/Downloads/Student folder/student_test"
$env:PATHSNAP_MODEL = (Resolve-Path artifacts/person2/hgb_full_train.joblib).Path
$env:PATHSNAP_EVALUATE_REFERENCE = '1'
$env:OMP_NUM_THREADS = '2'
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

The verification used Python 3.14.6, scikit-learn 1.9.1, and artifact SHA256 `677A1EB7DFD1C7F6BEE486A8D20E70C81C5B2DA7E3E7BFD16DC2A57B34F6BDF0`. Port 8000 was already serving the synthetic Review 1 API, so it was not stopped. A bad configured artifact fails at startup; omitting the model selects the geometry baseline. Verify the prediction model name, not just health.

On the frontend machine, use `npm ci` in `frontend`, set `VITE_DEMO_MODE=false` and `VITE_API_URL=http://127.0.0.1:8001` in untracked `.env.local` for same-machine testing, and restart `npm run dev -- --host localhost --strictPort`. If Windows reports a locked Rolldown binary, stop the existing Vite process from its terminal before retrying the clean install.

For two laptops, use trusted LAN only: bind FastAPI to `0.0.0.0:8000` on Laptop 1, allow TCP 8000 from Laptop 2 on the private network, and use Laptop 1’s LAN IPv4 and port in Laptop 2's frontend URL. Keep the UI origin `http://localhost:5173`, or set the exact alternate origin via `PATHSNAP_EXTRA_ORIGINS`. Check health, trip catalog, prediction, browser rendering and server request logs from Laptop 2. Do not expose the API publicly. No two-laptop or reserved evaluator run was performed.

## Pipeline and contract

The loader reads only track observations, nodes and directed arcs for inference. Feature generation projects coordinates to metric space and produces candidate rows keyed by trip, observation and original arc index. The fitted model accepts, in order: distance_m, heading_diff_deg, speed_mps, segment_length_m, time_gap_s, projection_fraction, heading_valid, time_valid. Its classes were `[0, 1]`; scoring selects class 1 and merges probabilities one-to-one on the candidate keys. Inference rejects nonfinite or out-of-range scores. Both independent top-1 and Viterbi consume these scores. Directed network transitions connect eligible candidates; missing candidates, invalid time gaps or unreachable transitions can split decoding. Reference alignments are read afterward only for permitted evaluation.

`GET /health` returns status; `GET /trips` returns string IDs preserving zeros; `POST /predict` accepts a trip ID and returns GPS points, route geometry, ordered arc IDs, metrics and model name. Coordinates in JSON are latitude first. With multiple blocks the geometry covers only the longest contiguous road block by distance; arc IDs contain all decoded blocks. No artificial links are added. Route continuity uses eligible adjacent pairs, so 100% does not imply a single block. Reference-free accuracy and sequence similarity are null; reference geometry and labels are not exposed.

## Measured results

All ten local student-test trip IDs returned HTTP 200 using HistGradientBoostingClassifier + HMM/Viterbi. An unknown ID returned 404; POST preflight allowed the configured Vite origin. See `results/pathsnap_student_test_http.csv` for counts, timings and scores. HTTP timings include evaluation and transport. Six trips produced more than one block, including 25 blocks on 00000076. The browser automation timed out; no rendered UI verification is claimed.

```powershell
python -m backend.evaluate $env:PATHSNAP_TRIP_ROOT --model $env:PATHSNAP_MODEL --out results/hgb_integration_student_test.csv
python -m pytest -q
```

| Metric | Classifier top-1 | Viterbi |
|---|---:|---:|
| Pointwise accuracy, pooled usable labels | 88.45% | 93.23% |
| Continuity, pooled eligible pairs | 88.58% | 99.92% |
| Sequence similarity, unweighted mean | 42.39% | 73.29% |

The pooled figures reproduce the previously reported values to two decimals. The shared candidate miss rate was 0.235% of usable labels. The evaluator records combined inference time (0.131–3.489 seconds per trip), not separate classifier and decoder timings. Accuracy declined on 00000007 and 00000065 despite improved continuity; the student-test measurements are not reserved evaluator results.
