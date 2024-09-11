import time
import json
import numpy as np
import boto3

from data_collection import get_paper_data

def upload_s3_data(data, bucket_name, file_name):
    s3 = boto3.client('s3')
    s3.put_object(Body=data, Bucket=bucket_name, Key=file_name)
    
    
def data_ingestion(
    query: str,
    api_key: str, 
    bucket_name: str, 
    total_records: int, 
    max_records: int = 25, 
    file_name: str = "raw_pdf_data.json"):
    data = []
    for i in np.arange(i, total_records, max_records):
        data += get_paper_data(query=query, api_key=api_key, starting_record=i, max_records=max_records)
        
        if total_records > 1200:
            time.sleep(1.2)
        
    json_data = json.dumps(data)
    upload_s3_data(data=json_data, bucket_name=bucket_name, file_name=file_name)
    
