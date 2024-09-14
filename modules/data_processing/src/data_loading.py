import json
import boto3
from typing import List, Dict

from ...utils.logger import setup_logger

logger = setup_logger("data_loading", "data_processing.log")

def load_data_from_s3(
    bucket_name: str, 
    file_name: str = 'vegan_research_papers.json'
) -> List[Dict]:
    """
    Loads JSON data from an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        file_name (str, optional): The name of the file object in the S3 bucket. Defaults to 'vegan_research_papers.json'.

    Returns:
        List[Dict]: The loaded JSON data.
    """
    logger.info(f"Loading data from S3 bucket: {bucket_name}, file: {file_name}")
    
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=file_name)

    status_code = response['ResponseMetadata']['HTTPStatusCode']
    if status_code != 200:
        raise RuntimeError(
            f"Failed to load data from S3, status code: {status_code}"
        )
    
    data = json.loads(response['Body'].read())
    
    return data


