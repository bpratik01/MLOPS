import os
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
import yaml
from typing import Tuple
from logger import setup_logger  # Assuming setup_logger is defined in the logger module

# Setup logger
logger = setup_logger("ModelBuilding", "logs/model_building.log")


def load_params(params_path: str) -> dict:
    """Load the model building parameters from the YAML configuration."""
    logger.info(f"Loading parameters from {params_path}")
    params = yaml.safe_load(open(params_path))["model_building"]
    logger.info("Parameters loaded successfully")
    return params


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data from a CSV file."""
    logger.info(f"Loading training data from {data_path}")
    train_data = pd.read_csv(data_path)
    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    y_train = pd.to_numeric(y_train, errors="coerce")
    
    # Ensure no missing values in target variable
    assert not np.any(pd.isna(y_train)), "y_train contains invalid or missing values."
    logger.info(f"Training data loaded with shape {train_data.shape}")
    return x_train, y_train


def train_model(x_train: np.ndarray, y_train: np.ndarray, params: dict) -> XGBClassifier:
    """Train the XGBoost model using the specified parameters."""
    logger.info(f"Training model with parameters: {params}")
    clf = XGBClassifier(
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        use_label_encoder=params["use_label_encoder"],
        eval_metric=params["eval_metric"]
    )
    clf.fit(x_train, y_train)
    logger.info("Model training completed successfully")
    return clf


def save_model(model: XGBClassifier, model_path: str) -> None:
    """Save the trained model to the specified path."""
    logger.info(f"Saving model to {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model, open(model_path, "wb"))
    logger.info(f"Model saved to {model_path}")


def main():
    logger.info("Starting model building pipeline")
    
    params_path = "params.yaml"
    data_path = "data/features/train_bow.csv"
    model_path = "models/xgboost_model.pkl"

    params = load_params(params_path)
    x_train, y_train = load_data(data_path)
    clf = train_model(x_train, y_train, params)
    save_model(clf, model_path)
    
    logger.info("Model building pipeline completed successfully")


if __name__ == "__main__":
    main()
