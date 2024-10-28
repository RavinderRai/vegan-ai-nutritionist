import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class AWSConfig:
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    SAGEMAKER_ROLE = os.environ.get("SAGEMAKER_ROLE")
    HUGGINGFACE_ACCESS_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

class S3Config:
    BUCKET_NAME = "fine-tuning-training-data"
    TRAIN_DATA_KEY = "train_data.json"
    TEST_DATA_KEY = "test_data.json"
    
    @classmethod
    def get_train_data_uri(cls):
        return f"s3://{cls.BUCKET_NAME}/{cls.TRAIN_DATA_KEY}"
    
    @classmethod
    def get_test_data_uri(cls):
        return f"s3://{cls.BUCKET_NAME}/{cls.TEST_DATA_KEY}"

class ModelConfig:
    MODEL_NAME = "tiiuae/falcon-7b-instruct"
    MAX_LENGTH = 512