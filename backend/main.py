"""FastAPI endpoints for synthetic and real PathSnap trips."""

import os

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, JsonValue

from .demo import ARC_ENDPOINTS, GPS_POINTS, TRIP_ID, demo_arc_ids
from .evaluation import evaluate_prediction
from .inference import TripStore, predict_trip
from .scoring import GeometryScorer, ModelScorer


class PredictionRequest(BaseModel):
    trip_id: str = Field(examples=[TRIP_ID])


class PredictionResponse(BaseModel):
    trip_id: str
    gps_points: list[tuple[float, float]]
    predicted_route: list[tuple[float, float]]
    predicted_arc_ids: list[int]
    metrics: dict[str, JsonValue] | None
    model_name: str


def _demo_response():
    arc_ids = demo_arc_ids()
    return {
        "trip_id": TRIP_ID,
        "gps_points": GPS_POINTS,
        "predicted_route": [ARC_ENDPOINTS[arc_ids[0]][0], *(ARC_ENDPOINTS[arc][1] for arc in arc_ids)],
        "predicted_arc_ids": arc_ids,
        "metrics": None,
        "model_name": "Synthetic Viterbi demo",
    }


def create_app(store=None, scorer=None, evaluate_reference=False, extra_origins=()):
    app = FastAPI(title="PathSnap", description="Map matching and route reconstruction")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", *extra_origins],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    scorer = scorer if scorer is not None else GeometryScorer()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/trips")
    async def trips() -> list[str]:
        return store.trip_ids() if store is not None else [TRIP_ID]

    def real_response(directory):
        prediction = predict_trip(directory, scorer)
        metrics = evaluate_prediction(prediction, directory if evaluate_reference else None)
        return PredictionResponse(**prediction.response(metrics))

    @app.post("/predict")
    async def predict(request: PredictionRequest) -> PredictionResponse:
        if store is None and request.trip_id == TRIP_ID:
            return PredictionResponse(**_demo_response())
        if store is not None:
            try:
                directory = store.directory(request.trip_id)
            except KeyError:
                pass
            else:
                return await run_in_threadpool(real_response, directory)
        raise HTTPException(status_code=404, detail="Unknown trip ID.")

    return app


def configured_app():
    root = os.environ.get("PATHSNAP_TRIP_ROOT")
    model = os.environ.get("PATHSNAP_MODEL")
    origins = os.environ.get("PATHSNAP_EXTRA_ORIGINS", "").split(",")
    return create_app(
        store=TripStore(root) if root else None,
        scorer=ModelScorer(model) if model else None,
        evaluate_reference=os.environ.get("PATHSNAP_EVALUATE_REFERENCE") == "1",
        extra_origins=[origin.strip() for origin in origins if origin.strip()],
    )


app = configured_app()
