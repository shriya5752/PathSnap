# PathSnap Review 1

This local demo uses synthetic candidate scores for trip `00000001`. At the middle GPS observation, the classifier prefers arc X (0.65) over arc B (0.35). The road graph allows A → B → C, so Viterbi uses B to complete the route. The trip ID, GPS points, and arc IDs in this demo belong to the synthetic fixture.

## Run the standalone decoder

From this folder in PowerShell, run:

```powershell
python demo.py
```

## Run the API

On Windows, double-click `Start_Review.cmd` to print the decoder comparison and start the API in the same window. Run `python -m pip install -r requirements.txt` if FastAPI or Uvicorn are missing. To start the server manually from this folder:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open [API docs](http://127.0.0.1:8000/docs), or run these requests in a second PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/trips

Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post `
  -ContentType 'application/json' -Body '{"trip_id":"00000001"}' |
  ConvertTo-Json
```

## Frontend API contract

CORS allows the Vite origin `http://localhost:5173`.

| Request | Response |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `GET /trips` | JSON array of trip ID strings, currently `["00000001"]`. |
| `POST /predict` | Accepts `{"trip_id":"00000001"}`. Returns the fields in the example below. Unknown trip IDs return HTTP 404. |

```json
{
  "trip_id": "00000001",
  "gps_points": [[12.97008, 77.59016], [12.97032, 77.59043], [12.97036, 77.59077]],
  "predicted_route": [[12.9701, 77.5901], [12.9702, 77.5903], [12.9703, 77.5906], [12.9704, 77.59085]],
  "predicted_arc_ids": [104, 205, 310],
  "metrics": null,
  "model_name": "Synthetic Viterbi demo"
}
```

Coordinate pairs are `[latitude, longitude]`. Arc IDs are integers preserved from the input graph. The endpoint runs Viterbi on each request and constructs the map line from the decoded arcs. The demo uses synthetic scores, performs no evaluation, and reads no `.route` data. For real trips, use `.route` only in permitted server-side evaluation after prediction.

## Explain it during the review

The candidate road segments are the hidden states. The mock classifier supplies emission scores, and the toy road graph supplies transition probabilities. For each candidate at each step, Viterbi keeps the best score and remembers which previous candidate led to it. It traces those choices backward to recover the best valid route. The decoder adds log scores, which is equivalent to multiplying the original positive scores.
