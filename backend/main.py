"""FastAPI endpoints for the PathSnap Review 1 demonstration."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, JsonValue

from demo import ARC_ENDPOINTS, GPS_POINTS, TRIP_ID, run_demo

app = FastAPI(title="PathSnap Review 1", description="Synthetic Viterbi demonstration")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    trip_id: str = Field(examples=[TRIP_ID])


class PredictionResponse(BaseModel):
    trip_id: str
    gps_points: list[tuple[float, float]]
    predicted_route: list[tuple[float, float]]
    predicted_arc_ids: list[int]
    metrics: dict[str, JsonValue] | None
    model_name: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/trips")
async def trips() -> list[str]:
    return [TRIP_ID]


@app.post("/predict")
async def predict(request: PredictionRequest) -> PredictionResponse:
    if request.trip_id != TRIP_ID:
        raise HTTPException(status_code=404, detail=f"Use synthetic trip ID {TRIP_ID}.")
    arc_ids = run_demo()["predicted_arc_ids"]
    route = [ARC_ENDPOINTS[arc_ids[0]][0]]
    route.extend(ARC_ENDPOINTS[arc_id][1] for arc_id in arc_ids)
    return PredictionResponse(
        trip_id=request.trip_id,
        gps_points=list(GPS_POINTS),
        predicted_route=route,
        predicted_arc_ids=arc_ids,
        metrics=None,
        model_name="Synthetic Viterbi demo",
    )
