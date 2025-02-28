import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils.data_transformation import DataTransformation

@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({
        'num_feature1': [1.0, 2.0, 3.0],
        'num_feature2': [4.0, 5.0, 6.0],
        'cat_feature': ['A', 'B', 'A']
    })
    X_test = pd.DataFrame({
        'num_feature1': [7.0, 8.0, 9.0],
        'num_feature2': [10.0, 11.0, 12.0],
        'cat_feature': ['B', 'A', 'B']
    })
    y_train = np.array([1, 2, 3])
    y_test = np.array([4, 5, 6])
    return X_train, X_test, y_train, y_test

@pytest.fixture
def data_transformation(tmp_path):
    feature_preprocessor_path = tmp_path / "feature_preprocessor.pkl"
    target_preprocessor_path = tmp_path / "target_preprocessor.pkl"
    return DataTransformation(feature_preprocessor_path, target_preprocessor_path)

def test_create_feature_preprocessor(data_transformation):
    num_columns = ['num_feature1', 'num_feature2']
    cat_columns = ['cat_feature']
    preprocessor = data_transformation.create_feature_preprocessor(num_columns, cat_columns)
    assert isinstance(preprocessor, ColumnTransformer)

def test_create_target_preprocessor():
    scaler = DataTransformation.create_target_preprocessor()
    assert isinstance(scaler, StandardScaler)

def test_transform_data(data_transformation, sample_data):
    X_train, X_test, y_train, y_test = sample_data
    transformed_X_train, transformed_X_test, transformed_y_train, transformed_y_test = data_transformation.transform_data(X_train, X_test, y_train, y_test)
    
    assert transformed_X_train.shape == (3, 4)  # 2 numerical + 2 one-hot encoded categories
    assert transformed_X_test.shape == (3, 4)
    assert transformed_y_train.shape == (3, 1)
    assert transformed_y_test.shape == (3, 1)

def test_save_preprocessor_objects(data_transformation, sample_data, tmp_path):
    X_train, X_test, y_train, y_test = sample_data
    data_transformation.transform_data(X_train, X_test, y_train, y_test)
    
    assert (tmp_path / "feature_preprocessor.pkl").exists()
    assert (tmp_path / "target_preprocessor.pkl").exists()
