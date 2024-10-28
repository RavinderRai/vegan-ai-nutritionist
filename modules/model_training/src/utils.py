import boto3
import sagemaker
import json
from .config import AWSConfig

class AWSConnector:
    def __init__(self):
        self.boto3_session = boto3.Session(
            aws_access_key_id=AWSConfig.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWSConfig.AWS_SECRET_ACCESS_KEY
        )
        
        self.sagemaker_session = self._init_sagemaker_session()
        self.s3_client = self.boto3_session.client('s3')
    
    def _init_sagemaker_session(self):
        sess = sagemaker.Session(boto_session=self.boto3_session)
        sagemaker_session_bucket = sess.default_bucket()
        return sagemaker.Session(default_bucket=sagemaker_session_bucket)
    
    def load_data_from_s3(self, bucket, key):
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read().decode('utf-8')
        return json.loads(data)
    
    def get_session_info(self):
        return {
            'role_arn': AWSConfig.SAGEMAKER_ROLE,
            'bucket': self.sagemaker_session.default_bucket(),
            'region': self.sagemaker_session.boto_region_name
        }
