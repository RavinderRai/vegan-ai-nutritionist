import mlflow
from datetime import datetime
from sagemaker.huggingface import get_huggingface_llm_image_uri

from ...config import AWSConfigCredentials, ModelDeploymentConfig
from ...utils.utils import AWSConnector
from ...utils.logger import setup_logger
from .model_utils import get_model_data_uri
from .model_deployment import deploy_model, deploy_base_model

logger = setup_logger("inference", "inference.log")

def save_endpoint_info(model_name: str, endpoint_name: str) -> None:
    """
    Save the endpoint name and timestamp to MLflow as parameters and artifacts.
    
    Args:
        endpoint_name (str): Name of the deployed endpoint
    """
    # Start an MLflow run
    mlflow.set_tracking_uri("sqlite:///C:/Users/RaviB/GitHub/vegan-ai-nutritionist/mlflow/endpoints.db")
    with mlflow.start_run() as run:
        # Log the endpoint name and timestamp as parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("endpoint_name", endpoint_name)
        mlflow.log_param("deployment_timestamp", str(datetime.now()))


    logger.info(f"Endpoint information saved to MLflow with run ID: {run.info.run_id}")

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
        model_name, endpoint = deploy_model(
            aws_credentials.sagemaker_role, 
            llm_image, 
            s3_model_uri, 
            deployment_config, 
            aws_connector.sagemaker_session
        )
        endpoint_name = endpoint.endpoint_name
        
        logger.info("Model deployed successfully: %s", endpoint_name)
        save_endpoint_info(model_name, endpoint_name)
        
    except Exception as e:
        logger.error("Failed to deploy the model: %s", e)
        logger.info("Attempting to deploy the base model instead...")
        try:
            model_name, endpoint = deploy_base_model(
                aws_credentials.sagemaker_role, 
                llm_image, 
                aws_connector.sagemaker_session
            )
            
            endpoint_name = endpoint.endpoint_name
            logger.info("Base model deployed successfully: %s", endpoint_name)
            save_endpoint_info(model_name, endpoint_name)
        except Exception as base_e:
            logger.error("Failed to deploy the base model: %s", base_e)
            raise RuntimeError("Both model and base model deployment failed.") from base_e

if __name__ == "__main__":
    main()