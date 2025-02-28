import numpy as np
import pandas as pd
import joblib
from typing import List
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.logger import setup_logger

class DataTransformation:
    def __init__(self, feature_preprocessor_path, target_preprocessor_path):
        self.feature_preprocessor_path = feature_preprocessor_path
        self.target_preprocessor_path = target_preprocessor_path
        self.logger = setup_logger()
        

    def create_feature_preprocessor(self, num_columns : List[str], cat_columns: List[str]) -> ColumnTransformer:
        """  This function creates a ColumnTransformer pipeline for the given columns.
                Args: 
                    num_columns (List): List of numerical column names
                    cat_columns (List): List of categorical column names
                    
                Returns: 
                    ColumnTransformer: The ColumnTransformer pipeline for the given columns
                    """
        # Create num transformer 
        num_transformer = Pipeline(
            steps = [
                ("scaler", StandardScaler())
            ]
        )

        # create column  transformer
        col_transformer = Pipeline(
            steps = [
                ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("scaler", StandardScaler())
            ]
        )

        # Combine both transformers to create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_columns),
                ("cat", col_transformer, cat_columns)
            ]
        )

        return preprocessor
    
    @staticmethod
    def create_target_preprocessor():
        scaler = StandardScaler()
        return scaler

    def transform_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ This function transforms the given data using the preprocessor pipeline.
                Args: 
                    data (pd.DataFrame): The DataFrame containing the data.
                    preprocessor (ColumnTransformer): The preprocessor pipeline.
                
                Returns: 
                    X_train (np.ndarray): Features for the training set
                    X_test (np.ndarray): Features for the testing set
                    y_train (np.ndarray): Labels for the training set
                    y_test (np.ndarray): Labels for the testing set
                    """
        
        # Create the columns list
        num_columns = list(X_train.select_dtypes(include="number").columns)
        cat_columns = list(X_train.select_dtypes(exclude="number").columns)
        
        # get the preprocessors

        self.logger.info("Creating preprocessor objects")
        feature_preprocessor = self.create_feature_preprocessor(num_columns, cat_columns)
        target_preprocessor = self.create_target_preprocessor()

        # transform the data
        self.logger.info("Transforming data")
        transformed_X_train = np.array(feature_preprocessor.fit_transform(X_train))
        transformed_X_test = np.array(feature_preprocessor.transform(X_test))

        # transform target data
        transformed_y_train = target_preprocessor.fit_transform(y_train.reshape(-1, 1))
        transformed_y_test = target_preprocessor.transform(y_test.reshape(-1, 1))

        # Save the preprocessor objects
        self.save_preprocessor_objects(feature_preprocessor, target_preprocessor)

        return (transformed_X_train, transformed_X_test, transformed_y_train, transformed_y_test)

    def save_preprocessor_objects(self, feature_preprocessor, target_preprocessor):
        """ Saves the preprocessor object to the given path"""
        self.logger.info("Saving preprocessor objects")

        joblib.dump(feature_preprocessor, self.feature_preprocessor_path)
        joblib.dump(target_preprocessor, self.target_preprocessor_path)
