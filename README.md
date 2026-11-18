# PathSnap — GPS Map Matching and Route Reconstruction

**Team 067** · Machine-learning map matching · [Technical report and integration notes](report/integration.md)

PathSnap reconstructs a vehicle's likely route on a **directed road network** from noisy GPS observations. The pipeline generates nearby road-arc candidates, scores them with a supervised classifier, and uses **HMM/Viterbi decoding** to select a geographically consistent sequence. A FastAPI service exposes predictions to a React/Leaflet map interface.

> **Evaluation scope:** The repository contains recorded results for 80 training trips and 10 student-test trips. The student-test reference routes are used for offline evaluation, not as inputs to prediction. The deployed decoder uses the separately trained `artifacts/person2/hgb_full_train.joblib` model (`max_iter=100`, `max_leaf_nodes=31`); the Optuna-tuned HGB is a separate experiment.

## Project structure

| Path | Purpose |
| --- | --- |
| [`preprocessing/`](preprocessing/) | Trip loading, exploratory analysis, road candidates, eight features, weak labels |
| [`models/`](models/) | Five-model comparison, final training, Optuna tuning, evaluation and plots |
| [`backend/`](backend/) | FastAPI endpoints, model inference, HMM/Viterbi and routing |
| [`frontend/`](frontend/) | React/Vite and Leaflet route visualisation |
| [`visualizations/`](visualizations/) | Folium map generation |
| [`results/`](results/) | Recorded evaluation CSVs and generated plots |
| [`artifacts/person2/`](artifacts/person2/) | Fitted HGB model and metadata |
| [`tests/`](tests/) | Python tests |
| [`report/`](report/) | Technical integration notes |

## Reproduce the trained application (Windows PowerShell)

**Prerequisites:** Python with a compatible scikit-learn installation, Node.js/npm, and the supplied dataset extracted locally. The dataset is **not committed**; use the authorised `Student folder` with `train/` and `student_test/` trip directories.

**1. Start the backend** from the repository root:

```powershell
python -m pip install -r backend/requirements-model.txt
$env:PATHSNAP_TRIP_ROOT = 'C:\path\to\Student folder\student_test'
$env:PATHSNAP_MODEL = (Resolve-Path 'artifacts/person2/hgb_full_train.joblib').Path
# Only for authorised offline evaluation using reference routes:
$env:PATHSNAP_EVALUATE_REFERENCE = '1'
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**2. Start the live frontend** in a second terminal:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
```

Edit `frontend/.env.local`:

```dotenv
VITE_API_URL=http://localhost:8000
VITE_DEMO_MODE=false
```

Then run `npm run dev -- --host localhost --strictPort` from `frontend/` and open **http://localhost:5173**. Restart Vite after changing the environment file. The default `.env.example` enables synthetic browser demo mode, so **set `VITE_DEMO_MODE=false`** for real API inference.

**3. Verify the API**:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/trips
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body '{"trip_id":"00000052"}'
```

The trip ID must exist in your locally extracted dataset. API documentation is at **http://127.0.0.1:8000/docs**. If `PATHSNAP_MODEL` is unset, real-trip mode uses a geometry baseline rather than the trained HGB. If `PATHSNAP_TRIP_ROOT` is unset, the backend uses a synthetic review fixture.

For the **two-laptop demonstration**, follow [frontend/DEPLOYMENT.md](frontend/DEPLOYMENT.md). Use a trusted local network; do not expose the API to the public internet.

## Method

1. **Preprocessing:** read `.track`, `.nodes`, `.arcs`, and reference `.route` files; validate coordinates and road topology.
2. **Candidates and features:** generate up to ten nearby directed arcs per observation. Features: `distance_m`, `heading_diff_deg`, `speed_mps`, `segment_length_m`, `time_gap_s`, `projection_fraction`, `heading_valid`, `time_valid`.
3. **Weak supervision:** align observations monotonically to reference arcs for training; exclude ambiguous and distant matches. Reference routes are **not used during inference**.
4. **Classification:** compare Logistic Regression, KNN, Decision Tree, Random Forest, and Histogram Gradient Boosting using trip-grouped validation.
5. **Route decoding:** use classifier scores as HMM emissions and directed shortest-path feasibility/distance consistency as transitions. Missing candidates, large time gaps, or unreachable arcs split decoding blocks.
6. **Visualisation:** inspect GPS observations and reconstructed route in the live Leaflet UI; generate separate Folium visualisations with the supplied scripts.

## Recorded experimental results

**Classifier comparison** (recorded five-fold grouped CV and student-test metrics):

| Model | CV Macro F1 | Student-test Top-1 | Student-test Macro F1 | Student-test log loss |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8417 | 0.8730 | 0.8509 | 0.1284 |
| KNN | 0.8719 | 0.8458 | 0.8868 | 0.2806 |
| Decision Tree | 0.8874 | 0.8572 | 0.9028 | 0.2706 |
| Random Forest | 0.9006 | 0.8845 | 0.9155 | 0.0779 |
| Histogram Gradient Boosting | **0.9058** | 0.8825 | **0.9198** | **0.0713** |

**Classifier-only vs HGB + Viterbi** across ten student-test trips:

| Metric | Classifier only | Classifier + Viterbi |
| --- | ---: | ---: |
| Pooled pointwise accuracy | 88.45% | **93.23%** |
| Pooled route continuity | 88.58% | **99.92%** |
| Mean route sequence similarity | 42.39% | **73.29%** |

These are aggregate results; some individual trips degrade in pointwise accuracy. For example, trip `00000007` fell from 92.28% to 88.58%, while continuity increased from 90.10% to 100.00%.

**Bayesian tuning:** Optuna TPE, 20 trials, five parameters, maximising grouped-CV Macro F1. The best trial objective was 0.906262. Untuned versus tuned student-test Macro F1 was 0.919788 versus 0.920065; log loss was 0.071283 versus 0.071975. This is a **separate experiment** from the full-data model used in the API demonstration.

Recorded CSVs and plots: [results/person2](results/person2), [decoder comparison](results/hgb_student_test_viterbi.csv), and [HTTP integration](results/pathsnap_student_test_http.csv). See [integration details](report/integration.md) for evaluation definitions and reproducibility caveats.

## Testing and additional documentation

```powershell
python -m unittest discover -s tests
cd frontend
npm run lint
npm run build
node --test tests/api.test.js
```

For preprocessing and Folium generation, see [Person 1 handoff](preprocessing/HANDOFF.md) and [visualizations/create_maps.py](visualizations/create_maps.py). For backend behaviour, see [backend/README.md](backend/README.md); for UI setup, see [frontend/README.md](frontend/README.md).

**Data and integrity:** Raw GPS trip data and row-level derived GPS files are not publicly committed. Metrics above are drawn from recorded project outputs; do not treat UI screenshots as evidence of an independently verified two-laptop deployment.
