import boto3
import sagemaker
import json

# if not using streamlit, use the following import statement 
from ..config import AWSConfigCredentials 
# otherwise, use this one
#from config import AWSConfigCredentials

class AWSConnector:
    def __init__(self):
        aws_credentials = AWSConfigCredentials.load()

        self.boto3_session = boto3.Session(
            aws_access_key_id=aws_credentials.aws_access_key_id,
            aws_secret_access_key=aws_credentials.aws_secret_access_key
        )
        
        self.s3_client = self.boto3_session.client('s3')
        self.sagemaker_client = self.boto3_session.client('sagemaker')
        self.sagemaker_session = self._init_sagemaker_session()
    
    def _init_sagemaker_session(self):
        sess = sagemaker.Session(boto_session=self.boto3_session)
        sagemaker_session_bucket = sess.default_bucket()
        return sagemaker.Session(default_bucket=sagemaker_session_bucket)
    
    def load_data_from_s3(self, bucket, key):
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read().decode('utf-8')
        return json.loads(data)
    
    def delete_sagemaker_model(self, model_name):
        self.sagemaker_client.delete_model(ModelName=model_name)
    
    def delete_sagemaker_endpoint(self, endpoint_name):
        self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
