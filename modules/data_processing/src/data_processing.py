import os
from dotenv import load_dotenv, find_dotenv

from .data_loading import load_data_from_s3
from .data_transformer import get_full_data, convert_to_doc_format, chunk_doc
from .embeddings import get_bedrock_embeddings, generate_embeddings
from .vector_storage import opensearch_client, create_index, index_documents
from ...utils.logger import setup_logger

load_dotenv(find_dotenv())

OPENSEARCH_ENDPOINT = os.environ.get('OPENSEARCH_ENDPOINT')
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')
AWS_REGION = os.environ.get('AWS_REGION')
AWS_BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')

logger = setup_logger("data_processing", "data_processing.log")

bucket_name = os.environ.get('AWS_BUCKET_NAME')

def data_processing(bucket_name, index_name):
    data = load_data_from_s3(bucket_name)
    
    full_data = get_full_data(data)
    
    documents = convert_to_doc_format(full_data)
    
    chunked_docs = chunk_doc(documents)
    
    bedrock_embedding = get_bedrock_embeddings()
    
    embedded_docs = generate_embeddings(chunked_docs, bedrock_embedding)
    
    client = opensearch_client(
        OPENSEARCH_ENDPOINT, 
        AWS_ACCESS_KEY, 
        AWS_SECRET_KEY, 
        AWS_REGION
    )
    
    embedding_dimension = len(embedded_docs[0]['embedding'])
    
    create_index(client, index_name, embedding_dimension)
    
    number_of_documents_indexed = index_documents(client, index_name, embedded_docs)
    logger.info(f"Indexed {number_of_documents_indexed} documents into index '{index_name}'.")
    
def data_processing(bucket_name, index_name, opensearch_endpoint, aws_access_key, aws_secret_key, aws_region):
    # Set up the logger
    logger = setup_logger("data_processing", "data_processing.log")
    logger.info("Starting data processing pipeline.")

    # Load data from S3
    data = load_data_from_s3(bucket_name)
    logger.info(f"Loaded data from S3 bucket '{bucket_name}'.")

    # Transform data
    full_data = get_full_data(data)
    documents = convert_to_doc_format(full_data)
    chunked_docs = []
    for doc in documents:
        chunked_docs.extend(chunk_doc(doc))
    logger.info("Transformed and chunked the documents.")

    # Generate embeddings
    bedrock_embeddings = get_bedrock_embeddings()
    embedded_docs = generate_embeddings(chunked_docs, bedrock_embeddings)
    logger.info("Generated embeddings for the documents.")

    # Create OpenSearch client
    client = opensearch_client(
        opensearch_endpoint,
        aws_access_key,
        aws_secret_key,
        aws_region
    )
    logger.info("Created OpenSearch client.")

    # Determine embedding dimension
    embedding_dimension = len(embedded_docs[0]['embedding'])

    # Create index in OpenSearch
    create_index(client, index_name, embedding_dimension)
    logger.info(f"Created index '{index_name}' in OpenSearch.")

    # Index documents
    number_of_documents_indexed = index_documents(client, index_name, embedded_docs)
    logger.info(f"Indexed {number_of_documents_indexed} documents into index '{index_name}'.")
