import os
import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from src.utils.data_ingestion import DataIngestion

@patch("pandas.read_csv")
def test_load_data(mock_read_csv):
    mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    mock_read_csv.return_value = mock_df
    ingestion = DataIngestion("http://example.com/data.csv", "data.csv", "artifacts", 0.2)
    df = ingestion.load_data()
    mock_read_csv.assert_called_once_with("data.csv")
    pd.testing.assert_frame_equal(df, mock_df)

def test_split_data():
    data = pd.DataFrame({
        "Unnamed: 0": [0, 1, 2, 3],
        "Feature1": [10, 20, 30, 40],
        "Feature2": [50, 60, 70, 80],
        "Price": [100, 200, 300, 400]
    })
    ingestion = DataIngestion("http://example.com/data.csv", "data.csv", "artifacts", 0.5)
    X_train, X_test, y_train, y_test = ingestion.split_data(data)
    
    assert X_train.shape == (2, 2)
    assert X_test.shape == (2, 2)
    assert y_train.shape == (2,)
    assert y_test.shape == (2,)

if __name__ == "__main__":
    pytest.main()
