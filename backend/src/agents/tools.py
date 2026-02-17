"""
Custom tools for the LangChain agent system.

These tools provide the AI agents with capabilities to:
- Analyze sensor data
- Predict RUL (Remaining Useful Life)
- Get maintenance recommendations
- Search engine documentation
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from langchain.tools import tool
import joblib

# Global references to loaded models (set during initialization)
_models = {
    'rul_predictor': None,
    'state_classifier': None,
    'preprocessor': None,
    'feature_columns': None,
    'data_loader': None
}


def initialize_tools(model_dir: str, data_path: str, dataset_id: str = "FD001"):
    """
    Initialize tools with loaded models.
    
    Args:
        model_dir: Directory containing trained models
        data_path: Path to CMAPSSData
        dataset_id: Dataset identifier
    """
    from src.models.rul_predictor import RULPredictor
    from src.models.state_classifier import StateClassifier
    from src.data_processor import CMAPSSDataLoader
    
    # Load models
    _models['rul_predictor'] = RULPredictor.load(
        os.path.join(model_dir, f"rul_predictor_{dataset_id}.joblib")
    )
    _models['state_classifier'] = StateClassifier.load(
        os.path.join(model_dir, f"state_classifier_{dataset_id}.joblib")
    )
    
    # Load preprocessor
    preprocessor_data = joblib.load(
        os.path.join(model_dir, f"preprocessor_{dataset_id}.joblib")
    )
    _models['preprocessor'] = preprocessor_data['preprocessor']
    _models['feature_columns'] = preprocessor_data['feature_columns']
    
    # Initialize data loader
    _models['data_loader'] = CMAPSSDataLoader(data_path, dataset_id)


@tool
def analyze_sensor_data(unit_id: int, cycle: Optional[int] = None) -> str:
    """
    Analyze sensor data for a specific engine unit.
    
    Use this tool when the user asks about sensor readings, engine performance,
    or wants to understand the current state of a specific engine.
    
    Args:
        unit_id: The engine unit ID (1-100 for FD001)
        cycle: Optional specific cycle to analyze. If not provided, uses the latest cycle.
    
    Returns:
        A detailed analysis of the sensor readings for the specified engine.
    """
    if _models['data_loader'] is None:
        return "Error: Tools not initialized. Please ensure models are loaded."
    
    try:
        # Load test data to analyze
        test_df, _ = _models['data_loader'].load_test_data()
        
        # Filter for the specific unit
        unit_data = test_df[test_df['unit_id'] == unit_id]
        
        if unit_data.empty:
            return f"No data found for engine unit {unit_id}. Valid unit IDs are 1-{test_df['unit_id'].max()}."
        
        # Get specific cycle or latest
        if cycle is not None:
            cycle_data = unit_data[unit_data['cycle'] == cycle]
            if cycle_data.empty:
                max_cycle = unit_data['cycle'].max()
                return f"Cycle {cycle} not found for unit {unit_id}. Latest cycle is {max_cycle}."
        else:
            cycle_data = unit_data[unit_data['cycle'] == unit_data['cycle'].max()]
            cycle = cycle_data['cycle'].values[0]
        
        # Analyze key sensors
        sensor_cols = [c for c in cycle_data.columns if c.startswith('sensor_')]
        latest_readings = cycle_data[sensor_cols].iloc[0]
        
        # Calculate statistics compared to all data
        all_stats = test_df[sensor_cols].describe()
        
        # Identify anomalies (readings outside 2 std from mean)
        anomalies = []
        for sensor in sensor_cols:
            value = latest_readings[sensor]
            mean = all_stats.loc['mean', sensor]
            std = all_stats.loc['std', sensor]
            
            if std > 0:
                z_score = abs(value - mean) / std
                if z_score > 2:
                    direction = "high" if value > mean else "low"
                    anomalies.append(f"{sensor}: {value:.2f} ({direction}, z-score: {z_score:.2f})")
        
        analysis = f"""
## Engine Unit {unit_id} - Cycle {cycle} Analysis

### Key Sensor Readings:
- Temperature at HPC outlet (sensor_3): {latest_readings.get('sensor_3', 'N/A'):.2f}
- Physical core speed (sensor_9): {latest_readings.get('sensor_9', 'N/A'):.2f}
- Total pressure at HPC outlet (sensor_7): {latest_readings.get('sensor_7', 'N/A'):.2f}
- Corrected fan speed (sensor_13): {latest_readings.get('sensor_13', 'N/A'):.2f}

### Operational Context:
- Total cycles recorded: {unit_data['cycle'].max()}
- Current cycle: {cycle}

### Anomaly Detection:
"""
        
        if anomalies:
            analysis += "⚠️ **Anomalies detected:**\n"
            for a in anomalies[:5]:  # Limit to top 5
                analysis += f"- {a}\n"
        else:
            analysis += "✅ No significant anomalies detected in sensor readings.\n"
        
        return analysis.strip()
        
    except Exception as e:
        return f"Error analyzing sensor data: {str(e)}"


@tool
def predict_rul(unit_id: int) -> str:
    """
    Predict the Remaining Useful Life (RUL) for an engine unit.
    
    Use this tool when the user asks about how long an engine will last,
    remaining lifespan, or when maintenance will be needed.
    
    Args:
        unit_id: The engine unit ID to predict RUL for
    
    Returns:
        Prediction of remaining operational cycles before failure and engine state assessment.
    """
    if _models['rul_predictor'] is None:
        return "Error: Models not initialized. Please ensure models are loaded."
    
    try:
        from src.data_processor import FeatureEngineer
        
        # Load test data
        test_df, ground_truth = _models['data_loader'].load_test_data()
        
        # Get data for specific unit
        unit_data = test_df[test_df['unit_id'] == unit_id]
        
        if unit_data.empty:
            return f"No data found for engine unit {unit_id}."
        
        # Feature engineering
        feature_eng = FeatureEngineer()
        unit_data = feature_eng.create_features(unit_data)
        
        # Get last cycle data
        last_cycle = unit_data[unit_data['cycle'] == unit_data['cycle'].max()]
        
        # Prepare features
        available_features = [f for f in _models['feature_columns'] if f in last_cycle.columns]
        
        # Pad missing features with zeros
        feature_vector = np.zeros((1, len(_models['feature_columns'])))
        for i, f in enumerate(_models['feature_columns']):
            if f in last_cycle.columns:
                feature_vector[0, i] = last_cycle[f].values[0]
        
        # Scale features
        X = _models['preprocessor'].scaler.transform(feature_vector)
        
        # Predict RUL
        predicted_rul = _models['rul_predictor'].predict(X)[0]
        
        # Predict state
        predicted_state = _models['state_classifier'].predict(X)[0]
        state_probs = _models['state_classifier'].predict_proba(X)
        
        # Get actual RUL if available
        actual_rul = ground_truth.iloc[unit_id - 1] if unit_id <= len(ground_truth) else None
        
        # Format response
        response = f"""
## RUL Prediction for Engine Unit {unit_id}

### Prediction Results:
🔮 **Predicted RUL**: {predicted_rul:.0f} operational cycles
🏷️ **Engine State**: {predicted_state.upper()}

### State Probabilities:
"""
        for state, probs in state_probs.items():
            response += f"- {state.capitalize()}: {probs[0]*100:.1f}%\n"
        
        response += f"""
### Interpretation:
{_models['state_classifier'].get_state_description(predicted_state)}

### Current Cycle Information:
- Last recorded cycle: {last_cycle['cycle'].values[0]}
"""
        
        if actual_rul is not None:
            response += f"\n📊 **Ground Truth RUL**: {actual_rul} cycles (for model validation)"
        
        return response.strip()
        
    except Exception as e:
        return f"Error predicting RUL: {str(e)}"


@tool
def get_maintenance_recommendation(unit_id: int, predicted_rul: Optional[float] = None) -> str:
    """
    Get maintenance recommendations for an engine unit.
    
    Use this tool when the user asks about maintenance schedules, 
    what actions to take, or how to handle an engine's condition.
    
    Args:
        unit_id: The engine unit ID
        predicted_rul: Optional pre-calculated RUL value. If not provided, will calculate.
    
    Returns:
        Detailed maintenance recommendations based on engine state.
    """
    try:
        # If RUL not provided, get prediction first
        if predicted_rul is None:
            from src.data_processor import FeatureEngineer
            
            test_df, _ = _models['data_loader'].load_test_data()
            unit_data = test_df[test_df['unit_id'] == unit_id]
            
            if unit_data.empty:
                return f"No data found for engine unit {unit_id}."
            
            feature_eng = FeatureEngineer()
            unit_data = feature_eng.create_features(unit_data)
            last_cycle = unit_data[unit_data['cycle'] == unit_data['cycle'].max()]
            
            feature_vector = np.zeros((1, len(_models['feature_columns'])))
            for i, f in enumerate(_models['feature_columns']):
                if f in last_cycle.columns:
                    feature_vector[0, i] = last_cycle[f].values[0]
            
            X = _models['preprocessor'].scaler.transform(feature_vector)
            predicted_rul = _models['rul_predictor'].predict(X)[0]
        
        # Generate recommendations based on RUL
        if predicted_rul > 125:
            urgency = "LOW"
            recommendations = """
### ✅ Routine Monitoring Recommended

**Actions:**
1. Continue standard monitoring schedule
2. Review sensor trends weekly
3. Schedule next inspection in 50 cycles
4. No immediate maintenance required

**Focus Areas:**
- Monitor temperature sensors (sensor_3, sensor_4)
- Track core speed stability (sensor_9)
- Log any unusual vibrations
"""
        elif predicted_rul > 50:
            urgency = "MEDIUM"
            recommendations = f"""
### ⚠️ Proactive Maintenance Recommended

**Timeline:** Schedule maintenance within next {int(predicted_rul/2)} cycles

**Immediate Actions:**
1. Increase monitoring frequency to daily
2. Order replacement parts for HPC components
3. Schedule maintenance window
4. Brief maintenance crew on expected work scope

**Inspection Priorities:**
- High Pressure Compressor (HPC) - primary degradation source
- Fan blade erosion
- Combustor liner condition
- Turbine blade clearances

**Risk Mitigation:**
- Have backup engine available
- Prepare contingency flight schedules
"""
        else:
            urgency = "CRITICAL"
            recommendations = f"""
### 🚨 CRITICAL - Immediate Action Required

**Predicted RUL: {predicted_rul:.0f} cycles**

**IMMEDIATE ACTIONS:**
1. ⛔ Ground aircraft until inspection complete
2. 🔧 Initiate emergency maintenance protocol
3. 📋 Document all recent sensor anomalies
4. 👥 Assign priority maintenance team

**Required Inspections:**
- Borescope inspection of HPC stages
- Fan blade integrity check
- Combustor and turbine thermal inspection
- Oil analysis for metal particles

**Documentation:**
- Complete failure mode analysis
- Update maintenance log
- Prepare incident report if grounding exceeds 24 hours
"""
        
        response = f"""
## Maintenance Recommendations for Engine Unit {unit_id}

**Urgency Level:** {urgency}
**Predicted RUL:** {predicted_rul:.0f} cycles

{recommendations}
"""
        return response.strip()
        
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"


@tool
def list_available_engines() -> str:
    """
    List all available engine units in the dataset.
    
    Use this tool when the user asks what engines are available,
    or needs to know the valid unit IDs to query.
    
    Returns:
        List of available engine unit IDs with basic statistics.
    """
    try:
        test_df, rul_values = _models['data_loader'].load_test_data()
        
        # Get summary for each unit
        unit_summary = test_df.groupby('unit_id').agg({
            'cycle': ['min', 'max', 'count']
        }).reset_index()
        
        unit_summary.columns = ['unit_id', 'first_cycle', 'last_cycle', 'total_records']
        
        response = f"""
## Available Engine Units

**Dataset:** {_models['data_loader'].dataset_id}
**Total Units:** {len(unit_summary)}
**Total Records:** {len(test_df):,}

### Unit Summary (first 10):
| Unit ID | Cycles Recorded | Last Cycle |
|---------|-----------------|------------|
"""
        for _, row in unit_summary.head(10).iterrows():
            response += f"| {int(row['unit_id'])} | {int(row['total_records'])} | {int(row['last_cycle'])} |\n"
        
        if len(unit_summary) > 10:
            response += f"\n... and {len(unit_summary) - 10} more units.\n"
        
        response += "\nUse `predict_rul` or `analyze_sensor_data` with any unit_id from 1 to " + str(len(unit_summary))
        
        return response.strip()
        
    except Exception as e:
        return f"Error listing engines: {str(e)}"
    


# Export all tools
AVAILABLE_TOOLS = [
    analyze_sensor_data,
    predict_rul,
    get_maintenance_recommendation,
    list_available_engines
]
