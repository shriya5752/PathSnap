# Two-laptop review setup

Use a trusted LAN with both laptops on the same network. Do not forward port 8000 on the router or expose the review API on a public interface beyond this LAN. Keep the frontend bound to loopback on Laptop 2.

## Laptop 1: FastAPI and inference

1. From the repository root, install the backend dependencies and point the API at the extracted student-test trips. Set `PATHSNAP_MODEL` to the checked-in HistGradientBoosting artifact to use the trained pipeline. If it is unset, the real-trip API uses the geometry baseline. Leave `PATHSNAP_TRIP_ROOT` unset for the synthetic review demo.

   ```powershell
   python -m pip install -r backend/requirements-model.txt
   $env:PATHSNAP_TRIP_ROOT = 'C:\path\to\Student folder\student_test'
   $env:PATHSNAP_MODEL = (Resolve-Path 'artifacts/person2/hgb_full_train.joblib').Path
   $env:PATHSNAP_EVALUATE_REFERENCE = '1' # Optional for permitted student-test evaluation
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

2. Run `ipconfig` and use the IPv4 of the active Wi-Fi/Ethernet adapter (for example `192.168.1.23`), not `127.0.0.1` or `0.0.0.0`.
3. Ask Person 3 to allow the exact browser origin `http://localhost:5173` in FastAPI CORS with GET, POST, and `Content-Type`; middleware must handle the OPTIONS preflight. Use the frontend's actual origin if different. The backend IP is not the browser origin. The backend already allows `http://localhost:5173`.
4. If Windows Firewall blocks access, allow inbound TCP 8000 on the Private profile only, preferably scoped to Laptop 2's IP. Close the temporary rule and stop Uvicorn after the review. Do not disable the firewall.

## Laptop 2: React and Leaflet

```powershell
Test-NetConnection 192.168.1.23 -Port 8000
Invoke-RestMethod http://192.168.1.23:8000/health
Invoke-RestMethod http://192.168.1.23:8000/trips
cd frontend
npm ci
Copy-Item .env.example .env.local
```

Use the actual backend IPv4 in the commands above and edit `.env.local` (preserve an existing file):

```dotenv
VITE_API_URL=http://192.168.1.23:8000
VITE_DEMO_MODE=false
```

Run `npm run dev -- --host localhost --strictPort`. Open `http://localhost:5173`, select a permitted trip, reconstruct, and inspect the returned route and metrics. Use a known permitted trip from Person 3, not a guessed ID. Check `model_name` and identify whether the reconstruction used the geometry baseline or the trained classifier.

## Troubleshooting

- TCP fails: verify IP, Uvicorn binding, firewall scope, and Wi-Fi client isolation. If the venue blocks peer access, use a trusted private LAN or run both processes on one laptop.
- PowerShell works but the browser fails: check the exact CORS origin, console preflight errors, actual Vite port and env values. An HTTPS frontend cannot normally fetch a plain HTTP API.
- Blank basemap: verify internet access to OSM tiles separately from LAN connectivity. The vector layers are drawn independently.
- Rehearsal fallback: set `VITE_DEMO_MODE=true` and restart Vite. Announce that it is synthetic; the app never switches silently.
