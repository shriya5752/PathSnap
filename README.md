PathSnap — GPS Map Matching and Route Reconstruction
Team 067 · GPS map matching using candidate road segments, supervised classifiers, and HMM/Viterbi decoding.

PathSnap reconstructs the road route followed by a vehicle from noisy GPS observations. The pipeline reads GPS tracks and a directed road network, generates nearby candidate road arcs, scores each candidate using geometric and motion features, and applies Viterbi decoding to obtain a more continuous route. A FastAPI backend serves predictions to a React/Leaflet map interface.
Team
- Shruti Choudhary — PES1UG24AM271
- Shriya Vasundhara Singh — PES1UG24AM269
- Shristi Shukla — PES1UG24AM268
- Shriyadita Mohanty — PES1UG24AM270
Repository structure
Folder	Contents
preprocessing/	Dataset exploration, candidate generation, feature extraction, weak labels
models/	Five classifiers, cross-validation, Bayesian tuning, evaluations
backend/	FastAPI endpoints, model scoring, HMM/Viterbi, route reconstruction
frontend/	React/Vite/Leaflet interactive map
visualizations/	Map generation including Folium
results/	Recorded metrics and evaluation CSVs
artifacts/	Trained model artifacts, when available
tests/	Automated checks
report/	Technical documentation and submission report


Dataset and features
The supplied dataset contains .track GPS observations, .nodes road-network nodes, .arcs directed road segments, and .route reference arc sequences. Do not commit raw student data. Keep the extracted dataset outside the repository or in an ignored directory.
- Training: 80 trips, 208,417 GPS observations.
- Student test: 10 trips, 19,160 GPS observations.
- Candidate generation: up to 10 nearby road arcs per GPS observation.
- Eight model features: distance_m, heading_diff_deg, speed_mps, segment_length_m, time_gap_s, projection_fraction, heading_valid, time_valid.
- Weak labels are created through monotone reference-route alignment. Ambiguous and far observations are excluded from supervised learning.
Model comparison
Five models were evaluated using trip-grouped cross-validation. The following are recorded experimental results, not synthetic training curves.
Model	CV Macro F1	Student-test Top-1	Student-test Macro F1	Log loss
Logistic Regression	0.8417	0.8730	0.8509	0.1284
KNN	0.8719	0.8458	0.8868	0.2806
Decision Tree	0.8874	0.8572	0.9028	0.2706
Random Forest	0.9006	0.8845	0.9155	0.0779
Histogram Gradient Boosting	0.9058	0.8825	0.9198	0.0713


Bayesian optimisation: Optuna TPE, 20 trials, maximising five-fold grouped-CV Macro F1. Best recorded objective: 0.906262. The tuned experiment used max_iter=55, max_leaf_nodes=41, learning_rate=0.10011158, min_samples_leaf=20, and l2_regularization=0.00093236. Tuned student-test Top-1: 0.883002; Macro F1: 0.920065; log loss: 0.071975. The untuned comparison had 0.882504, 0.919788 and 0.071283, respectively.

Important: The decoder comparison and single-laptop demonstration used the separately trained artifacts/person2/hgb_full_train.joblib model, configured with max_iter=100 and max_leaf_nodes=31, not the Optuna-tuned model.
HMM/Viterbi results
Each hidden state is a candidate directed road arc. Classifier probabilities provide emission scores; directed shortest-path feasibility and disagreement between network travel distance and GPS displacement provide transition scores. Missing candidates, invalid time gaps, and disconnected transitions split decoding blocks.
Ten student-test trips	Classifier only	Classifier + Viterbi
Pooled pointwise accuracy	88.45%	93.23%
Pooled route continuity	88.58%	99.92%
Mean route sequence similarity	42.39%	73.29%


Not every trip improved: for trip 00000007, pointwise accuracy decreased from 92.28% to 88.58%, while continuity rose from 90.10% to 100.00%.
Local setup — Windows PowerShell
Use a compatible Python environment and Node.js supported by the Vite version in frontend/package.json.
1. Clone this repository and extract the supplied student dataset separately.
2. Install backend/model dependencies from the repository root:
python -m pip install -r backend/requirements-model.txt
$env:PATHSNAP_TRIP_ROOT = 'C:\path\to\Student folder\student_test'
$env:PATHSNAP_MODEL = (Resolve-Path 'artifacts/person2/hgb_full_train.joblib').Path
# Optional; only for permitted offline reference-based evaluation:
$env:PATHSNAP_EVALUATE_REFERENCE = '1'
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
The trained artifact must exist at the configured path. If PATHSNAP_MODEL is omitted, real-trip mode uses a geometry baseline instead of the trained classifier. If PATHSNAP_TRIP_ROOT is omitted, the backend serves a synthetic review fixture.
3. In another PowerShell window, start the UI:
cd frontend
npm ci
Copy-Item .env.example .env.local
Edit frontend/.env.local to contain:
VITE_DEMO_MODE=false
VITE_API_URL=http://localhost:8000
Then run:
npm run dev -- --host localhost --strictPort
Open http://localhost:5173 and select a student-test trip. The interface supports route reconstruction, GPS observations, a map overlay, and prediction metrics. The map uses Leaflet; the separate Folium generator is in visualizations/create_maps.py.
API
- GET /health — server status.
- GET /trips — available trip IDs.
- POST /predict — JSON request such as {"trip_id":"00000052"}; returns GPS points, predicted route geometry, arc IDs, model name, and available metrics.
API documentation is available at http://127.0.0.1:8000/docs while the backend runs. Reference-route information is for permitted server-side evaluation and is not returned as a prediction input.
Reproducibility and testing
python -m unittest discover -s tests
python -m pytest -q
cd frontend
npm run build
npm run lint
node --test tests/api.test.js
See [`backend/README.md`](backend/README.md), [`frontend/README.md`](frontend/README.md), [`frontend/DEPLOYMENT.md`](frontend/DEPLOYMENT.md), and [`report/integration.md`](report/integration.md) for detailed instructions and the scope of verified demonstrations. A successful same-machine API demo is not, by itself, proof of a two-laptop deployment.
Evidence and submission
Keep the final technical report in report/ and labelled screenshots in report/screenshots/ (create this folder if needed). Include the actual Folium map with GPS observations, road network, reference route, and predicted route; UI screenshots; and successful API logs. Avoid posting private student data or unverified evaluation claims.
