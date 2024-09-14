import json
import boto3
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

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
    logger.info("Data loaded successfully from S3.")
    return data


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    
    AWS_BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')
    
    data = load_data_from_s3(AWS_BUCKET_NAME)
    
    print(data[0])
    