import json
import logging
import boto3
from ...config import AWSConfigCredentials
from ...utils.utils import AWSConnector

def load_endpoint_info(file_path: str = "endpoint_info.json") -> str:
    """
    Load the endpoint name from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing endpoint information.
    
    Returns:
        str: The name of the endpoint.
    """
    with open(file_path, 'r') as f:
        info = json.load(f)
    return info.get("endpoint_name")

def delete_endpoint(endpoint_name: str, sagemaker_session) -> None:
    """
    Delete a SageMaker endpoint.
    
    Args:
        endpoint_name (str): The name of the endpoint to delete.
        sagemaker_session: The SageMaker session.
    """
    try:
        sagemaker_client = sagemaker_session.boto_session.client('sagemaker')
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logging.info(f"Endpoint '{endpoint_name}' deleted successfully.")
    except Exception as e:
        logging.error(f"Failed to delete endpoint '{endpoint_name}': {e}")
        raise

def main():
    """
    Main function to delete a SageMaker endpoint using the name stored in a JSON file.
    """
    logging.basicConfig(level=logging.INFO)
    
    aws_credentials = AWSConfigCredentials.load()
    aws_connector = AWSConnector()
    
    # Load the endpoint name from the JSON file
    endpoint_name = load_endpoint_info()
    
    if endpoint_name:
        logging.info(f"Attempting to delete endpoint: {endpoint_name}")
        delete_endpoint(endpoint_name, aws_connector.sagemaker_session)
    else:
        logging.error("No endpoint name found in the JSON file.")

if __name__ == "__main__":
    main()