import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

load_dotenv(find_dotenv())

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_openai_client(open_api_key=OPENAI_API_KEY):
    gpt_client = OpenAI(api_key=open_api_key)
    return gpt_client

def get_opensearch_client(
    aws_access_key=AWS_ACCESS_KEY,
    aws_secret_key=AWS_SECRET_KEY,
    aws_region=AWS_REGION,
    opensearch_endpoint=OPENSEARCH_ENDPOINT
):
    awsauth = AWS4Auth(aws_access_key, aws_secret_key, aws_region, 'es')
    client = OpenSearch(
        hosts=[{'host': opensearch_endpoint, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    return client

