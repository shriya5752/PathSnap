# PathSnap — The Route Atlas

PathSnap compares noisy GPS observations with reconstructed route geometry in an editorial cartography interface. This frontend uses React, Vite, JavaScript, Leaflet, React Leaflet, Lucide and CSS.

## Run locally

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
npm run dev -- --host localhost --strictPort
```

Open `http://localhost:5173`. Use a modern browser and a Node version supported by the installed Vite version. If `.env.local` already exists, edit it rather than copying over it.

| Variable | Default | Use |
| --- | --- | --- |
| `VITE_DEMO_MODE` | `true` | `true` uses three synthetic browser fixtures; `false` uses only the backend. |
| `VITE_API_URL` | `http://localhost:8000` | API base URL, not a secret. |

Restart Vite after changing environment variables. A deployed static build must be rebuilt with the intended settings. Demo mode does not call the inference API or report evaluation scores. Live API mode denotes the transport, not proof that the model or returned trip is real; the checked-in FastAPI Review 1 app itself uses a synthetic fixture.

## API contract

- `GET /health` → `{"status":"ok"}`.
- `GET /trips` → a JSON array of string IDs (current backend), or `{"trip_ids":[...]}`.
- `POST /predict` ← `{"trip_id":"00000001"}`; returns `trip_id`, `gps_points`, `predicted_route`, `predicted_arc_ids`, `model_name` and optional or null `metrics`.
- Coordinates are `[latitude, longitude]`. `predicted_route` can be a flat sequence or an array of independent sequences. No line is drawn between pieces. Pieces with fewer than two coordinates do not form a line.
- Arc IDs may be strings or safe JSON integers. Use strings for IDs wider than JavaScript's safe integer range. The matched segment count is the number of distinct returned arc IDs, not the number of vertices.
- `metrics.accuracy` and `metrics.route_continuity` are displayed as supplied (finite numbers or display strings), without guessing units or rescaling. Absent values show `N/A`. Use explicit display strings if units are needed.
- Invalid responses, timeouts and HTTP errors are surfaced to the user. Live errors never fall back to fixtures. The client sends a trip ID only, never reference routes.

## Map and checks

Terracotta circles are GPS observations; teal lines are API-provided geometry in live mode. Select Raw GPS, Reconstructed, or Overlay. Click a GPS circle for coordinates. OSM tiles require internet; overlays still render without a basemap. Demo previews appear on selection; live GPS points arrive with prediction because the current API has no observation-preview endpoint. No classifier-only layer is shown because the current API does not supply its geometry.

Run `npm run build`, `npm run lint`, and `node --test tests/api.test.js`. In a browser, select trips, reconstruct, toggle layers, inspect a point, and switch trips during reconstruction. Confirm metrics show `N/A` when absent and check a narrow viewport. In API mode, test empty geometry and a stopped or unreachable backend; the interface should show an error rather than demo results. See `DEPLOYMENT.md` for LAN setup.

Geometry fidelity depends on the backend: polylines are not proof of road-network alignment, especially if the response contains endpoints only.
