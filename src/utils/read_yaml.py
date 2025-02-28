import yaml

def read_config(config_file_path, section):
    """
    Reads a specific section from a YAML configuration file.

    Parameters:
    config_file_path (str): The path to the YAML configuration file.
    section (str): The section of the configuration to read.

    Returns:
    dict: The configuration data for the specified section.
    """
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config.get(section)

# Example usage
config_file_path = 'src/config/config.yml'
section = 'data_ingestion_config'
data_ingestion_config = read_config(config_file_path, section)

print(data_ingestion_config)