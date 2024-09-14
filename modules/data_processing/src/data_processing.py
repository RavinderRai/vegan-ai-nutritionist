import os
import logging
from dotenv import load_dotenv, find_dotenv

from .config import INDEX_NAME
from .data_loading import load_data_from_s3
from .data_transformer import get_full_data, convert_to_doc_format, chunk_doc
from .embeddings import get_bedrock_embeddings, generate_embeddings
from .vector_storage import opensearch_client, create_index, index_documents
from ...utils.logger import setup_logger


def data_processing(
    INDEX_NAME: str,
    AWS_BUCKET_NAME: str,
    OPENSEARCH_ENDPOINT: str,
    AWS_ACCESS_KEY: str,
    AWS_SECRET_KEY: str,
    AWS_REGION: str
):
    logger = setup_logger("data_processing", "data_processing.log")
    logger.info("Starting main data processing function.")
    
    data = load_data_from_s3(AWS_BUCKET_NAME)
    logger.info("Data loaded successfully from S3.")
    
    full_data = get_full_data(data)
    logger.info("Full data aggregation complete.")
    
    documents = convert_to_doc_format(full_data)
    logger.info("Conversion to Langchain Document format complete.")
    
    chunked_docs = chunk_doc(documents)
    logger.info("Chunking documents complete.")

    logger.info("Initializing Bedrock embedding...")
    bedrock_embedding = get_bedrock_embeddings()
    
    logger.info("Generating embeddings...")
    embedded_docs = generate_embeddings(chunked_docs, bedrock_embedding)
    
    client = opensearch_client(
        OPENSEARCH_ENDPOINT, 
        AWS_ACCESS_KEY, 
        AWS_SECRET_KEY, 
        AWS_REGION
    )
    
    embedding_dimension = len(embedded_docs[0]['embedding'])
    
    create_index(client, INDEX_NAME, embedding_dimension)
    logger.info(f"Index '{INDEX_NAME}' created successfully.")
    
    number_of_documents_indexed = index_documents(client, INDEX_NAME, embedded_docs)
    logger.info(f"Indexed {number_of_documents_indexed} documents into index '{INDEX_NAME}'.")

    logger.info("Finished data processing pipeline.")

    
def main():
    # Load environment variables from .env file
    """
    Main entry point of the data processing pipeline.

    Loads environment variables from .env file and validates that all required variables are set.
    Then starts the data processing pipeline by calling data_processing function.

    Raises ValueError if any of the required environment variables are not set.
    """
    
    load_dotenv(find_dotenv())

    AWS_BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')
    OPENSEARCH_ENDPOINT = os.environ.get('OPENSEARCH_ENDPOINT')
    AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION')

    # Validate environment variables
    missing_vars = []
    if not AWS_BUCKET_NAME:
        missing_vars.append('AWS_BUCKET_NAME')
    if not OPENSEARCH_ENDPOINT:
        missing_vars.append('OPENSEARCH_ENDPOINT')
    if not AWS_ACCESS_KEY:
        missing_vars.append('AWS_ACCESS_KEY')
    if not AWS_SECRET_KEY:
        missing_vars.append('AWS_SECRET_KEY')
    if not AWS_REGION:
        missing_vars.append('AWS_REGION')
    if missing_vars:
        logger.error(f"The following environment variables are not set: {', '.join(missing_vars)}")
        raise ValueError(f"The following environment variables are not set: {', '.join(missing_vars)}")

    # Start the data processing pipeline
    data_processing(
        INDEX_NAME,
        AWS_BUCKET_NAME,
        OPENSEARCH_ENDPOINT,
        AWS_ACCESS_KEY,
        AWS_SECRET_KEY,
        AWS_REGION
    )

if __name__ == '__main__':
    main()