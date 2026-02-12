"""
Training script for RUL predictor and state classifier models.
Includes MLFlow experiment tracking for ML lifecycle management.

Usage:
    python -m src.train_models --data-path ./CMAPSSData --dataset FD001
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processor import prepare_training_data
from src.models.rul_predictor import RULPredictor
from src.models.state_classifier import StateClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MLFlow setup
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLFlow not installed. Run: pip install mlflow")


def train_models(
    data_path: str,
    dataset_id: str = "FD001",
    model_dir: str = "./models"
) -> dict:
    """
    Train both RUL predictor and state classifier models.
    
    Args:
        data_path: Path to CMAPSSData directory
        dataset_id: Dataset to use (FD001-FD004)
        model_dir: Directory to save trained models
        
    Returns:
        Dictionary with training results and metrics
    """
    logger.info(f"Starting model training for dataset {dataset_id}")
    
    # Set up MLFlow experiment
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(f"PredictiveMaintenance_{dataset_id}")
        logger.info("MLFlow tracking enabled")
    
    # Prepare data
    logger.info("Loading and preprocessing data...")
    data = prepare_training_data(data_path, dataset_id)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # ===== RUL Predictor Training =====
    rul_params = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    }
    
    logger.info("Training RUL Predictor...")
    start_time = time.time()
    
    rul_predictor = RULPredictor(**rul_params)
    rul_predictor.train(X_train, y_train)
    rul_train_time = time.time() - start_time
    
    # Evaluate RUL Predictor
    rul_metrics = rul_predictor.evaluate(X_test, y_test)
    logger.info(f"RUL Predictor Metrics:")
    logger.info(f"  RMSE: {rul_metrics['rmse']:.2f} cycles")
    logger.info(f"  MAE: {rul_metrics['mae']:.2f} cycles")
    logger.info(f"  R²: {rul_metrics['r2']:.4f}")
    logger.info(f"  S-Score: {rul_metrics['s_score']:.2f}")
    logger.info(f"  Training time: {rul_train_time:.2f}s")
    
    # Log RUL model to MLFlow
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"RUL_Predictor_{dataset_id}"):
            mlflow.log_params(rul_params)
            mlflow.log_params({
                "dataset": dataset_id,
                "model_type": "RandomForest_Regressor",
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "n_features": X_train.shape[1]
            })
            mlflow.log_metrics({
                "rmse": rul_metrics['rmse'],
                "mae": rul_metrics['mae'],
                "r2": rul_metrics['r2'],
                "s_score": rul_metrics['s_score'],
                "training_time_sec": rul_train_time
            })
            mlflow.sklearn.log_model(rul_predictor.model, "rul_predictor")
            logger.info("RUL Predictor logged to MLFlow")
    
    # ===== State Classifier Training =====
    state_params = {
        "n_estimators": 100,
        "max_depth": 12
    }
    
    logger.info("Training State Classifier...")
    start_time = time.time()
    
    state_classifier = StateClassifier(**state_params)
    state_classifier.train(X_train, y_train)
    state_train_time = time.time() - start_time
    
    # Evaluate State Classifier
    state_metrics = state_classifier.evaluate(X_test, y_test)
    logger.info(f"State Classifier Metrics:")
    logger.info(f"  Accuracy: {state_metrics['accuracy']:.4f}")
    logger.info(f"  Training time: {state_train_time:.2f}s")
    
    # Log State Classifier to MLFlow
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"State_Classifier_{dataset_id}"):
            mlflow.log_params(state_params)
            mlflow.log_params({
                "dataset": dataset_id,
                "model_type": "RandomForest_Classifier",
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "n_features": X_train.shape[1]
            })
            mlflow.log_metrics({
                "accuracy": state_metrics['accuracy'],
                "training_time_sec": state_train_time
            })
            mlflow.sklearn.log_model(state_classifier.model, "state_classifier")
            logger.info("State Classifier logged to MLFlow")
    
    # Save models to disk
    os.makedirs(model_dir, exist_ok=True)
    
    rul_model_path = os.path.join(model_dir, f"rul_predictor_{dataset_id}.joblib")
    state_model_path = os.path.join(model_dir, f"state_classifier_{dataset_id}.joblib")
    preprocessor_path = os.path.join(model_dir, f"preprocessor_{dataset_id}.joblib")
    
    rul_predictor.save(rul_model_path)
    state_classifier.save(state_model_path)
    
    # Save preprocessor for inference
    import joblib
    joblib.dump({
        'preprocessor': data['preprocessor'],
        'feature_columns': data['feature_columns']
    }, preprocessor_path)
    
    logger.info(f"Models saved to {model_dir}")
    
    return {
        'rul_metrics': rul_metrics,
        'state_metrics': state_metrics,
        'model_paths': {
            'rul_predictor': rul_model_path,
            'state_classifier': state_model_path,
            'preprocessor': preprocessor_path
        }
    }


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train RUL predictor and state classifier models"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./CMAPSSData",
        help="Path to CMAPSSData directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Dataset to use for training"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Directory to save trained models"
    )
    
    args = parser.parse_args()
    
    results = train_models(
        data_path=args.data_path,
        dataset_id=args.dataset,
        model_dir=args.model_dir
    )
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"\nRUL Predictor Performance:")
    print(f"  RMSE: {results['rul_metrics']['rmse']:.2f} cycles")
    print(f"  MAE: {results['rul_metrics']['mae']:.2f} cycles")
    print(f"\nState Classifier Performance:")
    print(f"  Accuracy: {results['state_metrics']['accuracy']:.2%}")
    print(f"\nModels saved to: {args.model_dir}")
    
    if MLFLOW_AVAILABLE:
        print(f"\n📊 View MLFlow dashboard: mlflow ui --port 5000")
    else:
        print(f"\n⚠️  Install MLFlow for experiment tracking: pip install mlflow")


if __name__ == "__main__":
    main()
