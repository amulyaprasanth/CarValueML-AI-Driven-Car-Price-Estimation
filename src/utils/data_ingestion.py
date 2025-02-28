import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    """This class is used to ingest and preprocess data for machine learning models."""
    def load_data(self, filepath: str) -> pd.DataFrame:
        """ Loads the data from a specified CSV file into a pandas DataFrame.
        Args: 
            filepath (str): The path to the CSV file.
        Returns:
            pd.DataFrame: A DataFrame containing the loaded data."""
        return pd.read_csv(filepath)


    @staticmethod
    def split_data(data: pd.DataFrame, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        # Split the data into features (X) and labels (y)
        features, labels = data.drop(["Price(k)"], axis=1), data["Price(k)"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

