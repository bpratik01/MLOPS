import os
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from logger import setup_logger  

logger = setup_logger("TextPreprocessing", "logs/text_preprocessing.log")

# Download necessary NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Paths for raw and processed data
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

# Text preprocessing functions
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """Apply all preprocessing steps to the text."""
    logger.debug("Starting text preprocessing")
    
    # Log basic info about the text before processing
    if len(text) > 50:
        logger.debug(f"Original text (first 50 chars): {text[:50]}...")
    else:
        logger.debug(f"Original text: {text}")

    # Lowercase the text
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    logger.debug("Removed URLs")

    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    logger.debug("Removed punctuation")

    # Remove numbers
    text = re.sub(r'\d+', '', text)
    logger.debug("Removed numbers")

    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    logger.debug("Removed stopwords")

    # Lemmatize words
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    logger.debug("Lemmatized words")

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    logger.debug("Removed extra whitespace")

    # Log processed text (first 50 characters)
    if len(text) > 50:
        logger.debug(f"Processed text (first 50 chars): {text[:50]}...")
    else:
        logger.debug(f"Processed text: {text}")

    return text

def normalize_data(df: pd.DataFrame, text_column: str = "content") -> pd.DataFrame:
    """Normalize the text content in the DataFrame."""
    logger.info(f"Normalizing data in column: {text_column}")
    
    # Apply text preprocessing
    df[text_column] = df[text_column].astype(str).apply(preprocess_text)
    
    # Remove short sentences
    df[text_column] = df[text_column].apply(lambda x: np.nan if len(x.split()) < 3 else x)
    df = df.dropna(subset=[text_column])  # Drop rows with NaN values

    logger.info(f"Normalization complete. Data shape: {df.shape}")
    return df

def save_data(df: pd.DataFrame, filename: str) -> None:
    """Save the DataFrame to the processed folder."""
    logger.info(f"Saving data to {filename}")
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_PATH, filename), index=False)
    logger.info(f"Data saved successfully to {filename}")

def main():
    logger.info("Starting text preprocessing")
    
    # Load raw data
    train_data = pd.read_csv(os.path.join(RAW_PATH, "train_data.csv"))
    test_data = pd.read_csv(os.path.join(RAW_PATH, "test_data.csv"))
    
    # Normalize data
    train_data_processed = normalize_data(train_data)
    test_data_processed = normalize_data(test_data)

    # Save processed data
    save_data(train_data_processed, "train_processed_data.csv")
    save_data(test_data_processed, "test_processed_data.csv")

    logger.info("Text preprocessing completed successfully")

if __name__ == "__main__":
    main()

    
