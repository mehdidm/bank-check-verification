from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import List, Dict
from . import api_auth  # Import the api_auth module

app = FastAPI()

# API key authentication
api_key_scheme = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Depends(api_key_scheme)):
    if not api_auth.validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key

async def get_api_key_with_permission(api_key: str = Depends(api_key_scheme), required_permission: str = None):
    if not api_auth.validate_api_key(api_key, required_permission):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key or insufficient permissions",
        )
    return api_key

# API Endpoints (stubs for now)

@app.post("/dataset/upload", dependencies=[Depends(get_api_key_with_permission(required_permission="upload"))])
async def upload_dataset():
    """Endpoint for uploading a dataset."""
    return {"message": "Dataset uploaded successfully"}

@app.post("/training/start", dependencies=[Depends(get_api_key_with_permission(required_permission="train"))])
async def start_training(model_name: str):
    """Endpoint for starting a training process."""
    return {"message": f"Training started for model: {model_name}"}

@app.get("/training/progress/{training_id}", dependencies=[Depends(get_api_key)])
async def get_training_progress(training_id: str):
    """Endpoint for getting the progress of a training process."""
    return {"training_id": training_id, "progress": 50}

@app.get("/models/compare", dependencies=[Depends(get_api_key)])
async def compare_models() -> List[Dict]:
    """Endpoint for comparing different models."""
    return [{"model": "YOLOv5", "mAP": 0.85}, {"model": "Faster R-CNN", "mAP": 0.82}]

@app.post("/models/deploy", dependencies=[Depends(get_api_key_with_permission(required_permission="deploy"))])
async def deploy_model(model_name: str):
    """Endpoint for deploying a trained model."""
    return {"message": f"Model {model_name} deployed successfully"}

# Example endpoint without authentication
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Check Processing API"}