import os
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

AWS_ACCESS_KEY_ID_ENV = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV = "AWS_SECRET_ACCESS_KEY"
SAGEMAKER_ROLE_ENV = "SAGEMAKER_ROLE"
HUGGINGFACE_ACCESS_TOKEN_ENV = "HUGGINGFACE_ACCESS_TOKEN"

@dataclass
class AWSConfigCredentials:
    aws_access_key_id: str
    aws_secret_access_key: str
    sagemaker_role: str
    huggingface_access_token: str

    @staticmethod
    def load():
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        sagemaker_role = os.getenv("SAGEMAKER_ROLE")
        huggingface_access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

        # Validate env variables' existence
        if not aws_access_key_id:
            raise ValueError("AWS_ACCESS_KEY_ID environment variable is not set.")
        if not aws_secret_access_key:
            raise ValueError("AWS_SECRET_ACCESS_KEY environment variable is not set.")
        if not sagemaker_role:
            raise ValueError("SAGEMAKER_ROLE environment variable is not set.")
        if not huggingface_access_token:
            raise ValueError("HUGGINGFACE_ACCESS_TOKEN environment variable is not set.")
        
        # Return an instance of the dataclass with validated values
        return AWSConfigCredentials(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            sagemaker_role=sagemaker_role,
            huggingface_access_token=huggingface_access_token,
        )

class ModelConfig:
    MODEL_NAME = "tiiuae/falcon-7b-instruct"
    MAX_LENGTH = 512

class ModelDeploymentConfig:
    INSTANCE_TYPE = "ml.g5.2xlarge"
    INITIAL_INSTANCE_COUNT = 1
    CONTAINER_STARTUP_HEALTH_CHECK_TIMEOUT = 300

    @staticmethod
    def get_deployment_config():
        return {
            "initial_instance_count": ModelDeploymentConfig.INITIAL_INSTANCE_COUNT,
            "instance_type": ModelDeploymentConfig.INSTANCE_TYPE,
            "container_startup_health_check_timeout": ModelDeploymentConfig.CONTAINER_STARTUP_HEALTH_CHECK_TIMEOUT,
        }

class S3Config:
    BUCKET_NAME = "fine-tuning-training-data"
    TRAIN_DATA_KEY = "train_data.json"
    TEST_DATA_KEY = "test_data.json"
    TOKENIZED_DATA_PATH = "tokenized"

    def __init__(self, model_name=ModelConfig.MODEL_NAME):
        self.model_name = model_name
    
    def get_train_data_uri(self):
        return f"s3://{self.BUCKET_NAME}/{self.TRAIN_DATA_KEY}"

    def get_test_data_uri(self):
        return f"s3://{self.BUCKET_NAME}/{self.TEST_DATA_KEY}"
    
    def get_tokenized_train_data_uri(self):
        return f"s3://{self.BUCKET_NAME}/{self.TOKENIZED_DATA_PATH}/{self.model_name}/tokenized-train-data"
    
    def get_tokenized_test_data_uri(self):
        return f"s3://{self.BUCKET_NAME}/{self.TOKENIZED_DATA_PATH}/{self.model_name}/tokenized-test-data"
