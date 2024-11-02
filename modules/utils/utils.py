import boto3
import sagemaker
import json
from ...config import AWSConfigCredentials

class AWSConnector:
    def __init__(self):
        aws_credentials = AWSConfigCredentials.load()

        self.boto3_session = boto3.Session(
            aws_access_key_id=aws_credentials.aws_access_key_id,
            aws_secret_access_key=aws_credentials.aws_secret_access_key
        )
        
        self.s3_client = self.boto3_session.client('s3')
        self.sagemaker_session = self._init_sagemaker_session()
    
    def _init_sagemaker_session(self):
        sess = sagemaker.Session(boto_session=self.boto3_session)
        sagemaker_session_bucket = sess.default_bucket()
        return sagemaker.Session(default_bucket=sagemaker_session_bucket)
    
    def load_data_from_s3(self, bucket, key):
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read().decode('utf-8')
        return json.loads(data)
