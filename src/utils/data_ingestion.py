import os
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import setup_logger


class DataIngestion:
    """This class is used to ingest and preprocess data for machine learning models."""
    def __init__(self, url: str, filepath: str, artifacts_dir: str, test_size: float) -> None:
        self.filepath = filepath
        self.test_size = test_size
        self.artifacts_dir = artifacts_dir
        self.url = url
        self.logger = setup_logger()

    def download_data(self):
        response = requests.get(self.url)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'wb') as f:
                self.logger.info("Downloading data from the URL...")
                f.write(response.content)

    def load_data(self) -> pd.DataFrame:
        """ Loads the data from a specified CSV file into a pandas DataFrame.
        Args: 
            filepath (str): The path to the CSV file.
        Returns:
            pd.DataFrame: A DataFrame containing the loaded data."""
        return pd.read_csv(self.filepath)


    def split_data(self, data: pd.DataFrame, ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """This function splits the given Dataframe into featuers and labels and then into trainng and testing sets
        Args:
            data (pd.DataFrame) : The DataFrame containing the data.
            test_size (float | Optional) : The proportion of the dataset to include in the test split. Defaults to 0.2.
        
        Returns:
        X_train (np.ndarray): Features for the training set
        X_test (np.ndarray): Features for the testing set
        y_train (np.ndarray): Labels for the training set
        y_test (np.ndarray): Labels for the testing set
        """

        # Drop the unnecessary column
        data = data.drop("Unnamed: 0", axis = 1)
        # Split the data into features (X) and labels (y)
        features, labels = data.drop(["Price"], axis=1), data["Price"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=self.test_size, random_state=42)

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


if __name__ == "__main__":
    from src.utils.read_yaml import read_config
    
    # read configuration from yaml file
    ingestion_config = read_config('config/config.yml', 'data_ingestion_config')

    data_ingestion = DataIngestion(**ingestion_config)
    data_ingestion.download_data()
    data = data_ingestion.load_data()

    X_train, X_test, y_train, y_test = data_ingestion.split_data(data)

    # Display the shapes
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")