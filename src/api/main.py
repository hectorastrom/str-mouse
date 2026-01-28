# @Time    : 2026-01-25 10:40
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : main.py

# Standalone FastAPI file to be hosted on AWS that serves the widget the best
# variants of our model
import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

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
from src.ml.utils import forward_pass, discover_all_checkpoints, FINETUNE_DIR


APP_DIR = os.path.dirname(__file__)
WIDGET_PATH = os.path.join(APP_DIR, "widget.html")

# Global model registry: maps num_classes -> loaded model
_MODELS: Dict[int, StrokeNet] = {}
_AVAILABLE_CHECKPOINTS: Dict[int, str] = {}  # num_classes -> checkpoint path
_DEFAULT_NUM_CLASSES: Optional[int] = None


def load_all_models() -> None:
    """Load all discovered checkpoint models into memory."""
    global _MODELS, _AVAILABLE_CHECKPOINTS, _DEFAULT_NUM_CLASSES
    
    _AVAILABLE_CHECKPOINTS = discover_all_checkpoints("finetune")
    
    if not _AVAILABLE_CHECKPOINTS:
        raise RuntimeError(
            f"No checkpoints found in {FINETUNE_DIR}. "
            "Expected folders like 'best_finetune-N-class' containing 'best.ckpt'."
        )
    
    # Set default to the highest class count
    _DEFAULT_NUM_CLASSES = max(_AVAILABLE_CHECKPOINTS.keys())
    
    for num_classes, ckpt_path in _AVAILABLE_CHECKPOINTS.items():
        print(f"Loading model: {ckpt_path} ({num_classes} classes)")
        model = StrokeNet.load_from_checkpoint(ckpt_path, num_classes=num_classes)
        model.eval()
        model.to(t.device("cpu"))
        _MODELS[num_classes] = model
    
    print(f"Loaded {len(_MODELS)} model(s). Default: {_DEFAULT_NUM_CLASSES} classes")


def get_model(num_classes: Optional[int] = None) -> StrokeNet:
    """
    Get a loaded model by class count.
    
    Args:
        num_classes: Number of classes for the model. If None, uses default (highest).
    
    Returns:
        The loaded StrokeNet model.
    
    Raises:
        KeyError: If no model with the requested class count is available.
    """
    if num_classes is None:
        num_classes = _DEFAULT_NUM_CLASSES
    
    if num_classes not in _MODELS:
        raise KeyError(f"No model available for {num_classes} classes")
    
    return _MODELS[num_classes]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload all models on startup to avoid cold start latency."""
    load_all_models()
    yield


app = FastAPI(
    title="scribble API",
    description="Decode mouse movements into characters using StrokeNet CNN",
    version="1.1.0",
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


class VelocityPoint(BaseModel):
    vx: float
    vy: float


class PredictRequest(BaseModel):
    velocities: List[VelocityPoint]
    num_classes: Optional[int] = None  # If None, uses default (highest available)


class PredictResponse(BaseModel):
    predicted_char: str
    confidence: float
    inference_ms: float
    image_base64: str
    image_data_url: str
    num_classes: int  # Which model variant was used


class ModelsResponse(BaseModel):
    available: List[int]  # List of available num_classes values
    default: int  # The default num_classes (highest)


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
    return {
        "status": "healthy",
        "models_loaded": len(_MODELS),
        "available_classes": sorted(_AVAILABLE_CHECKPOINTS.keys()),
    }


@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "name": "scribble API",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
    }


@app.get("/models", response_model=ModelsResponse)
def list_models():
    """
    List available model variants.
    
    Returns the available num_classes values and which one is the default.
    Use these values in the `num_classes` field of POST /predict requests.
    """
    return ModelsResponse(
        available=sorted(_AVAILABLE_CHECKPOINTS.keys()),
        default=_DEFAULT_NUM_CLASSES,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_letter(request: PredictRequest):
    """
    Predict a character from mouse velocity data.

    Request body:
    - velocities: List of velocity points [{vx, vy}, ...]
    - num_classes: (optional) Which model variant to use. Get available values
      from GET /models. If omitted, uses the default (highest class count).

    Response:
    - predicted_char: The predicted character (a-z, A-Z, 0-9, or space)
    - confidence: Prediction confidence as percentage (0-100)
    - inference_ms: Model inference time in milliseconds
    - image_base64: The rendered stroke image as base64 PNG
    - image_data_url: Ready-to-use data URL for <img> tags
    - num_classes: Which model variant was used for this prediction
    """
    velocities = parse_velocities_from_list(request.velocities)
    if velocities.size == 0:
        raise HTTPException(status_code=400, detail="No velocity samples provided.")

    # Validate and get the requested model
    num_classes = request.num_classes if request.num_classes is not None else _DEFAULT_NUM_CLASSES
    if num_classes not in _MODELS:
        available = sorted(_AVAILABLE_CHECKPOINTS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"No model available for {num_classes} classes. Available: {available}"
        )

    x_pos, y_pos = velocities_to_positions(velocities)
    x_tensor = t.from_numpy(x_pos).to(dtype=t.float32)
    y_tensor = t.from_numpy(y_pos).to(dtype=t.float32)
    img = build_img(x_tensor, y_tensor, invert_colors=False)

    model = get_model(num_classes)
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
        num_classes=num_classes,
    )


@app.get("/widget", response_class=HTMLResponse)
def serve_widget():
    """Serve the widget for local testing."""
    with open(WIDGET_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
