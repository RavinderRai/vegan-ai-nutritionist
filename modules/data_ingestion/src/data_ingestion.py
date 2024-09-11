import os
import time
import numpy as np
from .data_transformer import transform_paper_data
from .s3_storage import upload_to_s3
from dotenv import load_dotenv

load_dotenv()

def ingest_data(
    query: str,
    api_key: str, 
    bucket_name: str, 
    total_records: int,
    base_url: str = "http://api.springernature.com/openaccess/json",
    max_records: int = 25, 
    file_name: str = "raw_pdf_data.json"
) -> None:
    """
    Retrieves the meta data and full text content of papers that match the query from the Springer Nature API, 
    and uploads the data to an S3 bucket. 

    Args:
        query (str): Description of what kind of papers you want to collect.
        api_key (str): Springer Nature API Key. Note: there is a free version and a premium version.
        bucket_name (str): The name of the S3 bucket.
        total_records (int): The total number of records to collect.
        base_url (str, optional): The endpoint of the Springer Nature API to get meta data. 
            Defaults to "http://api.springernature.com/openaccess/json".
        max_records (int, optional): The max number of records you can query at a time. Defaults to 25.
            Will be set to 25 if input is higher (or lower than 1).
        file_name (str, optional): The name of the file object in the S3 bucket. Defaults to "raw_pdf_data.json".
    """
    all_data = []

    for i in np.arange(1, total_records, max_records):
        data_batch = transform_paper_data(query, api_key, base_url, i, max_records)
        all_data.extend(data_batch)
        
        # The transform_paper_data function makes 26 requests, but the rate limit is 100 per minute.
        # So we sleep for 20 seconds to limit to 3 functions calls per minute, thus avoiding rate limiting.
        if total_records > 78:  # Avoid rate limiting
            time.sleep(20)
    
    upload_to_s3(data=all_data, bucket_name=bucket_name, file_name=file_name)


if __name__ == "__main__":
    # Replace with appropriate values for your API key, bucket, etc.
    QUERY = "vegan nutrition"
    API_KEY = os.environ.get("SPRINGER_NATURE_API")
    BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')
    TOTAL_RECORDS = 100
    MAX_RECORDS = 25
    FILE_NAME = "raw_pdf_data.json"

    # Call the main ingestion function with the provided values
    ingest_data(
        query=QUERY, 
        api_key=API_KEY, 
        bucket_name=BUCKET_NAME, 
        total_records=TOTAL_RECORDS, 
        max_records=MAX_RECORDS, 
        file_name=FILE_NAME
    )