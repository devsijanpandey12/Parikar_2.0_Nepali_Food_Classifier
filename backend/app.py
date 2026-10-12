import json
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from models_handler import NepaliFoodModelManager, MODELS_METADATA

app = FastAPI(
    title="Parikar 2.0 - Nepali Food Classifier API",
    description="Next-generation AI classification API for authentic Nepali cuisine.",
    version="2.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model Manager
manager = NepaliFoodModelManager()

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
REPORTS_DIR = ROOT_DIR / "reports_and_charts"

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Parikar 2.0 Nepali Food Classifier",
        "models_count": len(MODELS_METADATA),
        "classes_count": len(manager.classes)
    }

@app.get("/api/models")
def get_models():
    """Returns list of available vision models and their architecture specs."""
    return {
        "models": manager.get_available_models(),
        "default_model": "convnext_v2"
    }

@app.get("/api/dishes")
def get_all_dishes():
    """Returns metadata for all supported Nepali food classes."""
    return {
        "classes": manager.classes,
        "dishes": manager.dishes
    }

@app.get("/api/dishes/{dish_id}")
def get_dish(dish_id: str):
    """Returns cultural, recipe, and dietary details for a specific dish."""
    dish = manager.get_dish_details(dish_id.lower())
    if not dish:
        raise HTTPException(status_code=404, detail=f"Dish '{dish_id}' not found.")
    return dish

@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns live empirical training metrics and chart image links for all models."""
    metrics_file = ROOT_DIR / "docs" / "training_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "status": "success",
                "models": data,
                "chart_url": "/reports/comparisons/model_comparison_latest.png",
                "dashboard_url": "/reports/comparisons/benchmark_dashboard.html"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "empty", "models": {}}

@app.post("/api/predict")
async def predict_food(
    image: UploadFile = File(...),
    model_id: str = Form("convnext_v2")
):
    """
    Classifies a food image using the selected vision model.
    Returns predicted class, confidence, top-3 candidates, and dish info.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")
    
    try:
        content = await image.read()
        result = manager.predict(content, model_id=model_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Serve static reports & charts if directory exists
REPORTS_DIR = ROOT_DIR / "reports_and_charts"
if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

# Serve static frontend files if directory exists.
# index.html references style.css / app.js / assets/* with root-relative paths, so the
# frontend must be mounted at "/". It is mounted last so the /api and /health routes
# above keep priority. "/static" is kept for backward compatibility.
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
