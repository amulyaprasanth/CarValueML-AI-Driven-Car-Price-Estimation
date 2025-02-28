import os
import hopsworks
from dotenv import load_dotenv
from src.logger import setup_logger

# Load .env file
load_dotenv()

# Load the environment variables
hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")

# Create a function to get feature store from hopsworks
def get_feature_store(api_key: str | None):
    """
    Retrieves the feature store from Hopsworks using the provided API key.

    This function attempts to log in to Hopsworks using the given API key
    and returns the feature store associated with the project.

    Args:
        api_key (str | None): The API key for Hopsworks authentication.
                              Can be None if not provided.

    Returns:
        FeatureStore | None: The feature store object if successful,
                             None if an error occurs during the process.

    Raises:
        Exception: Prints an error message if the feature store retrieval fails.
    """
    try:
        project = hopsworks.login(api_key_value=str(api_key))
        fs = project.get_feature_store()

        return fs

    except Exception as e:
        print(f"Failed to get feature store: {str(e)}")
        return None
