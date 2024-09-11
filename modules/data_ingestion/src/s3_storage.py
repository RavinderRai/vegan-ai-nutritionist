import json
import boto3
from typing import List, Dict

def upload_to_s3(data: List[Dict], bucket_name: str, file_name: str) -> None:
    """
    Uploads JSON data to an S3 bucket.

    Args:
        data (List[Dict]): The list of dictionaries to be uploaded to S3.
        bucket_name (str): The name of the S3 bucket.
        file_name (str): The name of the file object in the S3 bucket.
    """
    data = json.dumps(data)
    s3 = boto3.client('s3')
    s3.put_object(Body=data, Bucket=bucket_name, Key=file_name)
