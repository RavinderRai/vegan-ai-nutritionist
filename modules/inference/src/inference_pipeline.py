import json
from datetime import datetime
from model_utils import get_model_data_uri
from sagemaker.huggingface import get_huggingface_llm_image_uri

from ...config import AWSConfigCredentials, ModelDeploymentConfig
from ...utils.utils import AWSConnector
from ...utils.logger import setup_logger
from .model_deployment import deploy_model, deploy_base_model

logger = setup_logger("inference", "inference.log")

def save_endpoint_info(endpoint_name: str, file_path: str = "endpoint_info.json") -> None:
    """
    Save the endpoint name to a JSON file.
    
    Args:
        endpoint_name (str): Name of the deployed endpoint
        file_path (str): Path to save the endpoint information
    """
    info = {
        "endpoint_name": endpoint_name,
        "deployment_timestamp": str(datetime.datetime.now())
    }
    
    with open(file_path, 'w') as f:
        json.dump(info, f, indent=4)
    logger.info(f"Endpoint information saved to {file_path}")

def main():
    """
    Main function to deploy a Hugging Face model on SageMaker.
    
    This function attempts to deploy a specified Hugging Face model using SageMaker.
    If the deployment fails, it attempts to deploy a base model as a fallback.
    
    Steps:
    1. Load AWS connections and credentials.
    2. Load the model URI for SageMaker deployment.
    3. Retrieve the LLM image URI and deployment configuration.
    4. Attempt to deploy the model; if it fails, deploy the base model.
    
    Raises:
        RuntimeError: If both the model and base model deployments fail.
    """

    logger.info("Loading AWS connections...")
    aws_connector = AWSConnector()
    aws_credentials = AWSConfigCredentials.load()

    logger.info("Loading model uri for sagemaker deployment...")
    s3_model_uri = get_model_data_uri()['model_data_uri']
    
    
    logger.info("Retrieving LLM image URI and deployment configuration...")
    llm_image = get_huggingface_llm_image_uri("huggingface", version="1.3.3")
    deployment_config = ModelDeploymentConfig.get_deployment_config()

    try:
        logger.info("Deploying the model as a SageMaker endpoint...")
        llm = deploy_model(
            aws_credentials.sagemaker_role, 
            llm_image, 
            s3_model_uri, 
            deployment_config, 
            aws_connector.sagemaker_session
        )
        logger.info("Model deployed successfully: %s", llm)
        save_endpoint_info(llm.endpoint_name)
    except Exception as e:
        logger.error("Failed to deploy the model: %s", e)
        logger.info("Attempting to deploy the base model instead...")
        try:
            llm = deploy_base_model(
                aws_credentials.sagemaker_role, 
                llm_image, 
                aws_connector.sagemaker_session
            )
            logger.info("Base model deployed successfully: %s", llm)
            save_endpoint_info(llm.endpoint_name)
        except Exception as base_e:
            logger.error("Failed to deploy the base model: %s", base_e)
            raise RuntimeError("Both model and base model deployment failed.") from base_e

if __name__ == "__main__":
    main()