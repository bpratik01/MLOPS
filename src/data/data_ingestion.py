import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from logger import setup_logger

# Set up logger
logger = setup_logger("data_ingestion", "logs/data_ingestion.log")

def load_params(params_path: str) -> float:
  """Load the test size parameter from a YAML file."""
  logger.info(f"Loading parameters from {params_path}")
  with open(params_path, "r") as file:
    test_size = yaml.safe_load(file)["data_ingestion"]["test_size"]
  logger.info(f"Test size parameter loaded: {test_size}")
  return test_size


def read_data(url: str, raw_data_path: str) -> pd.DataFrame:
  """Read data from a URL and save it to the raw data folder."""
  logger.info(f"Reading data from {url}")
  df = pd.read_csv(url)
  os.makedirs(raw_data_path, exist_ok=True)
  df.to_csv(os.path.join(raw_data_path, "tweet_emotions.csv"), index=False)
  logger.info(f"Data saved to {raw_data_path}/tweet_emotions.csv")
  return df


def process_data(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Preprocess the data and split it into train and test datasets."""
  logger.info("Processing data and splitting into train/test sets")
  df.drop("tweet_id", axis=1, inplace=True)
  final_df = df[df["sentiment"].isin(["happiness", "sadness"])]
  final_df["sentiment"] = final_df["sentiment"].map({"happiness": 1, "sadness": 0})
  train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
  logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
  return train_data, test_data


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, raw_data_path: str) -> None:
  """Save the train and test datasets to the raw data folder."""
  logger.info("Saving train and test datasets")
  train_data.to_csv(os.path.join(raw_data_path, "train_data.csv"), index=False)
  test_data.to_csv(os.path.join(raw_data_path, "test_data.csv"), index=False)
  logger.info(f"Datasets saved to {raw_data_path}")


def main():
  logger.info("Starting data ingestion process")
  test_size = load_params("params.yaml")
  raw_data_path = "data/raw"

  df = read_data(
    "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv",
    raw_data_path,
  )
  train_data, test_data = process_data(df, test_size)
  save_data(train_data, test_data, raw_data_path)
  logger.info("Data ingestion completed successfully")


if __name__ == "__main__":
  main()
