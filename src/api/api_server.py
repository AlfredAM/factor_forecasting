"""
FastAPI REST API server for factor forecasting model inference
Provides endpoints for prediction, model management, and health monitoring
"""
import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
import asyncio
import threading
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_default_config
from src.models.models import create_model, load_model


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    factors: List[List[float]] = Field(..., description="Factor data for each time step")
    stock_ids: List[int] = Field(..., description="Stock IDs for each time step")
    dates: Optional[List[str]] = Field(None, description="Dates for each time step")
    
    class Config:
        schema_extra = {
            "example": {
                "factors": [[0.1, 0.2, 0.3, ...] for _ in range(20)],  # 20 time steps, 100 factors each
                "stock_ids": [1001, 1001, 1001, ...],  # 20 time steps
                "dates": ["2023-01-01", "2023-01-02", ...]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    predictions: Dict[str, List[float]] = Field(..., description="Predictions for each target")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")
    timestamp: str = Field(..., description="Prediction timestamp")


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    model_version: str
    model_size_mb: float
    total_parameters: int
    target_columns: List[str]
    num_factors: int
    max_seq_len: int
    device: str
    load_time: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    data: List[PredictionRequest] = Field(..., description="List of prediction requests")
    batch_size: Optional[int] = Field(32, description="Batch size for processing")


# Global variables for model management
model = None
config = None
model_info = None
model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting API server...")
    await load_model_async()
    yield
    # Shutdown
    logger.info("Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="Factor Forecasting API",
    description="REST API for factor forecasting model inference",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def load_model_async():
    """Load model asynchronously."""
    global model, config, model_info
    
    try:
        # Load configuration
        config = get_default_config()
        
        # Check for model path
        model_path = os.environ.get('MODEL_PATH', 'models/final_model.pth')
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, creating new model")
            model = create_model(config)
        else:
            logger.info(f"Loading model from {model_path}")
            model = load_model(model_path, config, device=getattr(config, 'device', 'cpu'))
        
        # Get model information
        model_info_dict = model.get_model_info()
        model_info = ModelInfo(
            model_name="FactorForecastingModel",
            model_version="1.0.0",
            model_size_mb=model_info_dict['model_size_mb'],
            total_parameters=model_info_dict['total_parameters'],
            target_columns=getattr(config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d']),
            num_factors=getattr(config, 'num_factors', 100),
            max_seq_len=getattr(config, 'max_seq_len', 50),
            device=str(model.device) if hasattr(model, 'device') else getattr(config, 'device', 'cpu'),
            load_time=datetime.now().isoformat()
        )
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def get_model():
    """Dependency to get the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def get_config():
    """Dependency to get the configuration."""
    if config is None:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    return config


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Factor Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Get memory usage
    memory_usage = {}
    if torch.cuda.is_available():
        memory_usage['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_usage['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
    
    import psutil
    memory_usage['cpu_memory_percent'] = psutil.virtual_memory().percent
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available(),
        memory_usage=memory_usage
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_info


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, model=Depends(get_model), config=Depends(get_config)):
    """
    Make predictions for a single sequence.
    
    Args:
        request: Prediction request containing factors and stock IDs
        model: Loaded model
        config: Model configuration
        
    Returns:
        Predictions for all targets
    """
    try:
        with model_lock:
            # Validate input
            if len(request.factors) != len(request.stock_ids):
                raise HTTPException(status_code=400, detail="Factors and stock_ids must have the same length")
            
            if len(request.factors) > config.get('max_seq_len', 50):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Sequence length {len(request.factors)} exceeds maximum {config.get('max_seq_len', 50)}"
                )
            
            # Convert to tensors
            factors = torch.tensor(request.factors, dtype=torch.float32)
            stock_ids = torch.tensor(request.stock_ids, dtype=torch.long)
            
            # Add batch dimension
            factors = factors.unsqueeze(0)  # (1, seq_len, num_factors)
            stock_ids = stock_ids.unsqueeze(0)  # (1, seq_len)
            
            # Move to device
            device = next(model.parameters()).device
            factors = factors.to(device)
            stock_ids = stock_ids.to(device)
            
            # Make prediction
            with torch.no_grad():
                predictions = model(factors, stock_ids)
            
            # Convert predictions to lists
            prediction_dict = {}
            for target, pred_tensor in predictions.items():
                prediction_dict[target] = pred_tensor.cpu().numpy().tolist()
            
            # Create metadata
            metadata = {
                "sequence_length": len(request.factors),
                "num_factors": len(request.factors[0]) if request.factors else 0,
                "unique_stocks": len(set(request.stock_ids)),
                "model_device": str(device),
                "prediction_time": datetime.now().isoformat()
            }
            
            return PredictionResponse(
                predictions=prediction_dict,
                metadata=metadata,
                timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest, model=Depends(get_model), config=Depends(get_config)):
    """
    Make predictions for multiple sequences in batch.
    
    Args:
        request: Batch prediction request
        model: Loaded model
        config: Model configuration
        
    Returns:
        List of predictions for each sequence
    """
    try:
        with model_lock:
            batch_size = request.batch_size or 32
            all_predictions = []
            
            # Process in batches
            for i in range(0, len(request.data), batch_size):
                batch_requests = request.data[i:i + batch_size]
                
                # Prepare batch data
                batch_factors = []
                batch_stock_ids = []
                batch_lengths = []
                
                for req in batch_requests:
                    if len(req.factors) != len(req.stock_ids):
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Request {i}: Factors and stock_ids must have the same length"
                        )
                    
                    batch_factors.append(req.factors)
                    batch_stock_ids.append(req.stock_ids)
                    batch_lengths.append(len(req.factors))
                
                # Pad sequences to same length
                max_len = max(batch_lengths)
                padded_factors = []
                padded_stock_ids = []
                
                for factors, stock_ids in zip(batch_factors, batch_stock_ids):
                    # Pad factors
                    if len(factors) < max_len:
                        padding = [[0.0] * len(factors[0]) for _ in range(max_len - len(factors))]
                        factors = factors + padding
                    
                    # Pad stock IDs
                    if len(stock_ids) < max_len:
                        padding = [stock_ids[-1] for _ in range(max_len - len(stock_ids))]
                        stock_ids = stock_ids + padding
                    
                    padded_factors.append(factors)
                    padded_stock_ids.append(stock_ids)
                
                # Convert to tensors
                factors = torch.tensor(padded_factors, dtype=torch.float32)
                stock_ids = torch.tensor(padded_stock_ids, dtype=torch.long)
                lengths = torch.tensor(batch_lengths, dtype=torch.long)
                
                # Move to device
                device = next(model.parameters()).device
                factors = factors.to(device)
                stock_ids = stock_ids.to(device)
                
                # Make predictions
                with torch.no_grad():
                    predictions = model(factors, stock_ids, lengths)
                
                # Process predictions for each sequence
                for j, req in enumerate(batch_requests):
                    prediction_dict = {}
                    for target, pred_tensor in predictions.items():
                        # Take only the valid predictions (not padded)
                        valid_pred = pred_tensor[j].cpu().numpy().tolist()
                        prediction_dict[target] = valid_pred
                    
                    metadata = {
                        "sequence_length": len(req.factors),
                        "num_factors": len(req.factors[0]) if req.factors else 0,
                        "unique_stocks": len(set(req.stock_ids)),
                        "model_device": str(device),
                        "prediction_time": datetime.now().isoformat()
                    }
                    
                    all_predictions.append(PredictionResponse(
                        predictions=prediction_dict,
                        metadata=metadata,
                        timestamp=datetime.now().isoformat()
                    ))
            
            return all_predictions
            
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """
    Reload the model in the background.
    
    Args:
        background_tasks: FastAPI background tasks
        
    Returns:
        Reload status
    """
    background_tasks.add_task(load_model_async)
    return {"message": "Model reload started", "timestamp": datetime.now().isoformat()}


@app.get("/model/status")
async def get_model_status():
    """Get current model status."""
    return {
        "model_loaded": model is not None,
        "config_loaded": config is not None,
        "model_info": model_info.dict() if model_info else None,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server.
    
    Args:
        host: Server host
        port: Server port
        reload: Whether to enable auto-reload
    """
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Factor Forecasting API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model file")
    
    args = parser.parse_args()
    
    # Set model path environment variable if provided
    if args.model_path:
        os.environ['MODEL_PATH'] = args.model_path
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    start_server(host=args.host, port=args.port, reload=args.reload) 