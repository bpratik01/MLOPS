import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from logger import setup_logger  


# Setup logger
logger = setup_logger("FeatureEngineering", "logs/feature_engineering.log")


def load_params(params_path: str) -> int:
    """Load the max_features parameter from the YAML configuration."""
    logger.info(f"Loading parameters from {params_path}")
    params = yaml.safe_load(open(params_path, "r"))
    max_features = params["feature_engineering"]["max_features"]
    logger.info(f"Max features loaded: {max_features}")
    return max_features


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and test data."""
    logger.info(f"Loading training data from {train_path}")
    logger.info(f"Loading test data from {test_path}")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    train_data.fillna("", inplace=True)
    test_data.fillna("", inplace=True)
    
    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")
    
    return train_data, test_data


def prepare_features(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    max_features: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare bag-of-words features for the training and test data."""
    logger.info(f"Preparing features with max_features={max_features}")
    
    vectorizer = CountVectorizer(max_features=max_features)
    
    # Extract features and labels
    X_train = train_data["content"].values
    y_train = train_data["sentiment"].astype(int).values
    X_test = test_data["content"].values
    y_test = test_data["sentiment"].astype(int).values
    
    # Fit and transform training data
    logger.info("Fitting and transforming training data")
    X_train_bow = vectorizer.fit_transform(X_train)
    logger.info("Transforming test data")
    X_test_bow = vectorizer.transform(X_test)
    
    return X_train_bow.toarray(), y_train, X_test_bow.toarray(), y_test


def save_features(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    output_dir: str
) -> None:
    """Save the processed features and labels to the specified directory."""
    logger.info(f"Saving features to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training features and labels
    train_df = pd.DataFrame(X_train)
    train_df["label"] = y_train
    train_df.to_csv(os.path.join(output_dir, "train_bow.csv"), index=False)
    logger.info(f"Training features saved to {output_dir}/train_bow.csv")
    
    # Save testing features and labels
    test_df = pd.DataFrame(X_test)
    test_df["label"] = y_test
    test_df.to_csv(os.path.join(output_dir, "test_bow.csv"), index=False)
    logger.info(f"Test features saved to {output_dir}/test_bow.csv")


def main():
    logger.info("Starting feature engineering pipeline")
    
    params_path = "params.yaml"
    train_path = "./data/processed/train_processed_data.csv"
    test_path = "./data/processed/test_processed_data.csv"
    output_dir = os.path.join("data", "features")
    
    # Pipeline steps
    max_features = load_params(params_path)
    train_data, test_data = load_data(train_path, test_path)
    X_train, y_train, X_test, y_test = prepare_features(train_data, test_data, max_features)
    save_features(X_train, y_train, X_test, y_test, output_dir)
    
    logger.info("Feature engineering pipeline completed successfully")


if __name__ == "__main__":
    main()
