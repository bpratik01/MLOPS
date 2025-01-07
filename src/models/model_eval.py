import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, Dict
from logger import setup_logger  

# Setup logger
logger = setup_logger("ModelEvaluation", "logs/model_evaluation.log")


def load_test_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the test data for evaluation."""
    logger.info(f"Loading test data from {data_path}")
    test_data = pd.read_csv(data_path)
    x_test = test_data.iloc[:, :-1].values
    y_test = pd.to_numeric(test_data.iloc[:, -1].values, errors="coerce")
    
    # Ensure no missing values in target variable
    assert not np.any(pd.isna(y_test)), "y_test contains invalid or missing values."
    logger.info(f"Test data loaded with shape {test_data.shape}")
    return x_test, y_test


def load_model(model_path: str) -> pickle:
    """Load the trained model."""
    logger.info(f"Loading model from {model_path}")
    model = pickle.load(open(model_path, "rb"))
    logger.info("Model loaded successfully")
    return model


def evaluate_model(
    clf, x_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate the model and calculate key metrics."""
    logger.info("Evaluating model")
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba[:, 1]),
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    logger.info(f"Saving metrics to {output_path}")
    with open(output_path, "w") as file:
        json.dump(metrics, file, indent=4)
    logger.info(f"Metrics saved to {output_path}")


def main():
    logger.info("Starting model evaluation pipeline")
    
    data_path = "data/features/test_bow.csv"
    model_path = "models/xgboost_model.pkl"
    metrics_path = "metrics.json"

    x_test, y_test = load_test_data(data_path)
    clf = load_model(model_path)
    metrics = evaluate_model(clf, x_test, y_test)
    save_metrics(metrics, metrics_path)
    
    logger.info("Model evaluation pipeline completed successfully")


if __name__ == "__main__":
    main()
