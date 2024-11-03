import logging

from .model_utils import get_model_data_uri
from ...utils.utils import AWSConnector


def main():
    """
    Main function to delete latest SageMaker endpoint using the name stored in a JSON file.
    """
    logging.basicConfig(level=logging.INFO)
    
    aws_connector = AWSConnector()
    
    # Get the endpoint name from the endpoints ssqlite database
    latest_endpoint = get_model_data_uri(['model_name', 'endpoint_name'], run_number=0, mlflow_data_path="sqlite:///mlflow/endpoints.db")
    model_name = latest_endpoint['model_name']
    endpoint_name = latest_endpoint['endpoint_name']
    
    if endpoint_name:
        logging.info(f"Deleting SageMaker model and endpoint: {endpoint_name}")
        aws_connector.delete_sagemaker_model(model_name) 
        aws_connector.delete_sagemaker_endpoint(endpoint_name)
        logging.info("SageMaker model and endpoint successfully deleted.")
    else:
        logging.info("No endpoint to delete.")
    
if __name__ == "__main__":
    main()