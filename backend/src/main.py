"""
FastAPI backend for the Predictive Maintenance Assistant.

Provides REST API endpoints for:
- RUL prediction
- Conversational chat interface
- Sensor data analysis
- Engine listing and health checks
"""

import os
import uuid
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from .schemas import (
    PredictRequest, PredictResponse,
    ChatRequest, ChatResponse,
    AnalyzeRequest, AnalyzeResponse,
    EngineListResponse, HealthResponse,
    MaintenanceResponse, ErrorResponse
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "assistant": None,
    "models_loaded": False,
    "conversations": {},
    "dataset_id": "FD001"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application."""
    logger.info("Starting Predictive Maintenance Assistant API...")
    
    # Load models on startup
    try:
        model_dir = os.getenv("MODEL_PATH", "./models")
        data_path = os.getenv("DATA_PATH", "./CMAPSSData")
        dataset_id = os.getenv("DATASET_ID", "FD001")
        
        app_state["dataset_id"] = dataset_id
        
        # Check if models exist
        if os.path.exists(os.path.join(model_dir, f"rul_predictor_{dataset_id}.joblib")):
            from .agents.graph import create_assistant
            app_state["assistant"] = create_assistant(model_dir, data_path, dataset_id)
            app_state["models_loaded"] = True
            logger.info(f"Models loaded successfully for dataset {dataset_id}")
        else:
            logger.warning(f"Models not found in {model_dir}. Run training first.")

            
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Turbofan Engine Predictive Maintenance API",
    description="""
    AI-powered predictive maintenance system for turbofan engines.
    
    Features:
    - **RUL Prediction**: Predict Remaining Useful Life in operational cycles
    - **State Classification**: Classify engine health (healthy/degrading/critical)
    - **Conversational AI**: Natural language interface powered by LangGraph
    - **Sensor Analysis**: Analyze sensor data and detect anomalies
    """,
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


# ============== Health & Info Endpoints ==============

@app.get("/api")
async def root():
    """API root - returns basic info."""
    return {
        "name": "Turbofan Engine Predictive Maintenance API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": app_state["models_loaded"]
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and model health status."""
    return HealthResponse(
        status="healthy" if app_state["models_loaded"] else "degraded",
        models_loaded=app_state["models_loaded"],
        dataset=app_state["dataset_id"],
        version="1.0.0"
    )


# ============== Prediction Endpoints ==============

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_rul(request: PredictRequest):
    """
    Predict Remaining Useful Life for an engine unit.
    
    Returns the predicted RUL in operational cycles along with
    state classification and confidence scores.
    """
    if not app_state["models_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run training first."
        )
    
    try:
        from .agents.tools import predict_rul as predict_tool
        
        # Call the prediction tool
        result = predict_tool.invoke({"unit_id": request.unit_id})
        
        # Parse the result (it's a formatted string from the tool)
        # In production, you'd want a more structured return
        
        # For now, return a structured response
        # This is a simplified example - in production, call models directly
        from .agents.tools import _models
        from .data_processor import FeatureEngineer, CMAPSSDataLoader
        import numpy as np
        
        loader = _models['data_loader']
        test_df, _ = loader.load_test_data()
        unit_data = test_df[test_df['unit_id'] == request.unit_id]
        
        if unit_data.empty:
            raise HTTPException(status_code=404, detail=f"Engine unit {request.unit_id} not found")
        
        # Process and predict
        feature_eng = FeatureEngineer()
        unit_data = feature_eng.create_features(unit_data)
        last_cycle = unit_data[unit_data['cycle'] == unit_data['cycle'].max()]
        
        feature_vector = np.zeros((1, len(_models['feature_columns'])))
        for i, f in enumerate(_models['feature_columns']):
            if f in last_cycle.columns:
                feature_vector[0, i] = last_cycle[f].values[0]
        
        X = _models['preprocessor'].scaler.transform(feature_vector)
        
        predicted_rul = float(_models['rul_predictor'].predict(X)[0])
        predicted_state = _models['state_classifier'].predict(X)[0]
        state_probs = _models['state_classifier'].predict_proba(X)
        
        # Determine confidence
        max_prob = max(p[0] for p in state_probs.values())
        confidence = "high" if max_prob > 0.7 else "medium" if max_prob > 0.5 else "low"
        
        return PredictResponse(
            unit_id=request.unit_id,
            predicted_rul=round(predicted_rul, 1),
            state=predicted_state,
            state_probabilities={k: round(float(v[0]), 3) for k, v in state_probs.items()},
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maintenance", response_model=MaintenanceResponse, tags=["Prediction"])
async def get_maintenance(request: PredictRequest):
    """
    Get maintenance recommendations for an engine unit.
    
    Returns urgency level and specific recommendations based on predicted RUL.
    """
    if not app_state["models_loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # First get the prediction
        prediction = await predict_rul(request)
        
        # Determine urgency and recommendations
        rul = prediction.predicted_rul
        
        if rul > 125:
            urgency = "LOW"
            recommendations = "Continue routine monitoring. Schedule next inspection in 50 cycles."
            next_inspection = 50
        elif rul > 50:
            urgency = "MEDIUM"
            recommendations = f"Schedule maintenance within {int(rul/2)} cycles. Monitor HPC sensors closely."
            next_inspection = int(rul / 2)
        else:
            urgency = "CRITICAL"
            recommendations = "Immediate inspection required. Ground aircraft for maintenance."
            next_inspection = 0
        
        return MaintenanceResponse(
            unit_id=request.unit_id,
            predicted_rul=rul,
            urgency=urgency,
            recommendations=recommendations,
            next_inspection_cycles=next_inspection
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Maintenance recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Chat Endpoint ==============

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Conversational interface with the predictive maintenance assistant.
    
    Supports natural language queries about engine health, predictions,
    and maintenance recommendations.
    """
    if not app_state["models_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run training first."
        )
    
    try:
        assistant = app_state["assistant"]
        
        # Get or create conversation
        conv_id = request.conversation_id or str(uuid.uuid4())
        history = app_state["conversations"].get(conv_id, [])
        
        # Get response from real Gemini API
        response = assistant.chat(request.message, history)
        
        # Determine if the response was successful (not a fallback error)
        is_success = "couldn't generate a text response" not in response.lower()
        
        # Update conversation history only on success
        if is_success:
            from langchain_core.messages import HumanMessage, AIMessage
            history.append(HumanMessage(content=request.message))
            history.append(AIMessage(content=response))
            app_state["conversations"][conv_id] = history[-20:]  # Keep last 20 messages
        
        return ChatResponse(
            response=response,
            conversation_id=conv_id,
            tools_used=[],  # Could extract from graph state
            success=is_success
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Chat error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Check if it's a Gemini API error
        error_str = str(e)
        if "NOT_FOUND" in error_str or "404" in error_str:
            return ChatResponse(
                response="""⚠️ **Chat Feature Unavailable**

The AI chat is currently unavailable due to Google API configuration issues.

**However, all core features work perfectly:** ✅
1. Select an engine from the dropdown above
2. Click "PREDICT RUL" to get predictions
3. View maintenance recommendations
4. Analyze engine health

**To enable chat:**
- Get a valid Gemini API key from: https://aistudio.google.com/app/apikey
- Update the GOOGLE_API_KEY in the .env file
- Restart the backend server

The predictive maintenance features don't need chat to work!""",
                conversation_id=request.conversation_id or str(uuid.uuid4()),
                tools_used=[],
                success=False
            )
        elif "RESOURCE_EXHAUSTED" in error_str or "429" in error_str or "quota" in error_str.lower():
            return ChatResponse(
                response="""⚠️ **API Quota Exceeded**

You've hit the Gemini API rate limit. This happens with free tier API keys.

**Quick fix:** Wait 1 minute and try again.

**Permanent fix:** The backend has been updated to use gemini-1.5-flash (1500 requests/day instead of 20).
**Restart the backend server** to apply the change:
```
Ctrl+C to stop
python -m uvicorn src.main:app --reload --port 8000
```

**All other features work fine:** ✅
- RUL Prediction
- Maintenance Recommendations  
- Engine Analysis""",
                conversation_id=request.conversation_id or str(uuid.uuid4()),
                tools_used=[],
                success=False
            )
        
        # For other errors, raise HTTP exception
        raise HTTPException(status_code=500, detail=str(e))





# ============== Analysis Endpoints ==============

@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_sensors(request: AnalyzeRequest):
    """
    Analyze sensor data for a specific engine and cycle.
    
    Returns detailed sensor analysis and detected anomalies.
    """
    if not app_state["models_loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        from .agents.tools import analyze_sensor_data
        
        params = {"unit_id": request.unit_id}
        if request.cycle:
            params["cycle"] = request.cycle
        
        analysis = analyze_sensor_data.invoke(params)
        
        return AnalyzeResponse(
            unit_id=request.unit_id,
            cycle=request.cycle or 0,
            analysis=analysis,
            anomalies=[]  # Could parse from analysis text
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/engines", response_model=EngineListResponse, tags=["Data"])
async def list_engines():
    """
    List all available engine units in the dataset.
    """
    if not app_state["models_loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        from .agents.tools import _models
        
        loader = _models['data_loader']
        test_df, _ = loader.load_test_data()
        
        unit_summary = test_df.groupby('unit_id').agg({
            'cycle': ['max', 'count']
        }).reset_index()
        unit_summary.columns = ['unit_id', 'max_cycle', 'records']
        
        engines = [
            {
                "unit_id": int(row['unit_id']),
                "max_cycle": int(row['max_cycle']),
                "records": int(row['records'])
            }
            for _, row in unit_summary.iterrows()
        ]
        
        return EngineListResponse(
            dataset=app_state["dataset_id"],
            total_engines=len(engines),
            engines=engines
        )
        
    except Exception as e:
        logger.error(f"List engines error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Training Endpoint ==============

@app.post("/train", tags=["Training"])
async def trigger_training(dataset_id: str = "FD001"):
    """
    Trigger model training (for development/setup).
    
    Note: This is a blocking operation and may take several minutes.
    """
    try:
        from .train_models import train_models
        
        data_path = os.getenv("DATA_PATH", "./CMAPSSData")
        model_dir = os.getenv("MODEL_PATH", "./models")
        
        results = train_models(data_path, dataset_id, model_dir)
        
        # Reload models
        from .agents.graph import create_assistant
        app_state["assistant"] = create_assistant(model_dir, data_path, dataset_id)
        app_state["models_loaded"] = True
        app_state["dataset_id"] = dataset_id
        
        return {
            "status": "success",
            "dataset": dataset_id,
            "metrics": {
                "rul_rmse": round(results['rul_metrics']['rmse'], 2),
                "rul_mae": round(results['rul_metrics']['mae'], 2),
                "state_accuracy": round(results['state_metrics']['accuracy'], 4)
            }
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Static File Serving ==============

# Path to the compiled frontend
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

# Serve the static files (Must be at the very end to avoid intercepting API calls)
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Root route to serve index.html
    @app.get("/", tags=["UI"])
    async def serve_frontend():
        return FileResponse(os.path.join(static_dir, "index.html"))
    
    # Catch-all route for Next.js routing
    @app.get("/{full_path:path}", tags=["UI"])
    async def entrypoint(full_path: str):
        file_path = os.path.join(static_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(static_dir, "index.html"))


# Run with: uvicorn src.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
