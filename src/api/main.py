import base64
import io
import os
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import torch as t
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from torchvision.transforms.functional import to_pil_image

from src.data.utils import (
    build_char_map,
    build_img,
    velocities_to_positions,
)
from src.ml.architectures.cnn import StrokeNet
from src.ml.utils import forward_pass

DEFAULT_CKPT_PATH = os.environ.get(
    "STROKNET_CKPT_PATH",
    "checkpoints/mouse_finetune/best_finetune/best.ckpt"
)
APP_DIR = os.path.dirname(__file__)
WIDGET_PATH = os.path.join(APP_DIR, "widget.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model on startup to avoid cold start latency."""
    get_model()
    yield


app = FastAPI(
    title="scribble API",
    description="Decode mouse movements into characters using StrokeNet CNN",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for cross-origin requests from blog
ALLOWED_ORIGINS = [
    "https://hectorastrom.com",
    "https://www.hectorastrom.com",
    "http://localhost:3000",  # local development
    "http://localhost:8000",  # local FastAPI dev
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

CHAR_MAP = build_char_map()
_MODEL = None


class VelocityPoint(BaseModel):
    vx: float
    vy: float


class PredictRequest(BaseModel):
    velocities: List[VelocityPoint]


class PredictResponse(BaseModel):
    predicted_char: str
    confidence: float
    inference_ms: float
    image_base64: str
    image_data_url: str


def get_model() -> StrokeNet:
    """Lazy-load and cache the StrokeNet model."""
    global _MODEL
    if _MODEL is None:
        model = StrokeNet.load_from_checkpoint(
            DEFAULT_CKPT_PATH,
            num_classes=53,
        )
        model.eval()
        model.to(t.device("cpu"))
        _MODEL = model
    return _MODEL


def parse_velocities_from_list(velocity_points: List[VelocityPoint]) -> np.ndarray:
    """Convert list of VelocityPoint objects to numpy array."""
    if not velocity_points:
        return np.array([], dtype=np.float64)
    return np.array(
        [(p.vx, p.vy) for p in velocity_points],
        dtype=np.float64
    )


def build_png_base64(img_tensor: t.Tensor) -> str:
    """Convert image tensor to base64-encoded PNG string."""
    pil_img = to_pil_image(img_tensor.unsqueeze(0))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "healthy", "model_loaded": _MODEL is not None}


@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "name": "scribble API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_letter(request: PredictRequest):
    """
    Predict a character from mouse velocity data.

    Accepts a list of velocity points (vx, vy) and returns:
    - predicted_char: The predicted character (a-z, A-Z, or space)
    - confidence: Prediction confidence as percentage (0-100)
    - inference_ms: Model inference time in milliseconds
    - image_base64: The rendered stroke image as base64 PNG
    - image_data_url: Ready-to-use data URL for <img> tags
    """
    velocities = parse_velocities_from_list(request.velocities)
    if velocities.size == 0:
        raise HTTPException(status_code=400, detail="No velocity samples provided.")

    x_pos, y_pos = velocities_to_positions(velocities)
    x_tensor = t.from_numpy(x_pos).to(dtype=t.float32)
    y_tensor = t.from_numpy(y_pos).to(dtype=t.float32)
    img = build_img(x_tensor, y_tensor, invert_colors=False)

    model = get_model()
    img_batch = img.unsqueeze(0).unsqueeze(0)
    char_probs, predicted_char, inference_time = forward_pass(
        model, img_batch, CHAR_MAP, log_time=True
    )
    confidence = float(char_probs.max().item()) * 100.0
    image_base64 = build_png_base64(img)

    return PredictResponse(
        predicted_char=predicted_char,
        confidence=confidence,
        inference_ms=inference_time * 1000.0,
        image_base64=image_base64,
        image_data_url=f"data:image/png;base64,{image_base64}",
    )


@app.get("/widget", response_class=HTMLResponse)
def serve_widget():
    """Serve the widget for local testing."""
    with open(WIDGET_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
