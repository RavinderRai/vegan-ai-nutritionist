import sagemaker
from sagemaker.huggingface import HuggingFaceModel

from ...config import ModelConfig, ModelDeploymentConfig

def deploy_huggingface_model(sagemaker_role: str, llm_image: str, sagemaker_session: sagemaker.Session, env: dict, s3_model_uri: str = None) -> sagemaker.Model:
    """
    Deploys a Hugging Face model to a SageMaker endpoint.

    Args:
    - sagemaker_role (str): The SageMaker execution role.
    - llm_image (str): The URI of the Hugging Face model image.
    - sagemaker_session (sagemaker.Session): The SageMaker session.
    - env (dict): A dictionary of environment variables for the model.
    - s3_model_uri (str, optional): The S3 URI of the model data.

    Returns:
    - sagemaker.Model: The deployed model.
    """
    model_kwargs = {
        "role": sagemaker_role,
        "image_uri": llm_image,
        "env": env,
        "sagemaker_session": sagemaker_session,
    }
    
    # Add model_data only if provided
    if s3_model_uri:
        model_kwargs["model_data"] = s3_model_uri

    hf_model = HuggingFaceModel(**model_kwargs)

    llm = hf_model.deploy(
        initial_instance_count=ModelDeploymentConfig.INITIAL_INSTANCE_COUNT,
        instance_type=ModelDeploymentConfig.INSTANCE_TYPE,
        container_startup_health_check_timeout=ModelDeploymentConfig.CONTAINER_STARTUP_HEALTH_CHECK_TIMEOUT
    )

    return llm

def deploy_model(sagemaker_role, llm_image, s3_model_uri, config, sagemaker_session):
    return deploy_huggingface_model(
        sagemaker_role=sagemaker_role,
        llm_image=llm_image,
        sagemaker_session=sagemaker_session,
        env=config,
        s3_model_uri=s3_model_uri
    )


def deploy_base_model(sagemaker_role, llm_image, sagemaker_session):
    hub = {'HF_MODEL_ID': ModelConfig.MODEL_NAME}
    return deploy_huggingface_model(
        sagemaker_role=sagemaker_role,
        llm_image=llm_image,
        sagemaker_session=sagemaker_session,
        env=hub
    )