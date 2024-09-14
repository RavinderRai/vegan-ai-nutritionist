from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from opensearchpy.exceptions import NotFoundError
from opensearchpy.helpers import bulk
import logging

from .config import INDEX_BODY

logger = logging.getLogger(__name__)

def opensearch_client(OPENSEARCH_ENDPOINT, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION):
    awsauth = AWS4Auth(AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, 'es')
    logger.info("Initializing OpenSearch client...")
    
    client = OpenSearch(
        hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    return client



def create_index(client, index_name, embedding_dimension):
    # Update the embedding dimension in the index body
    index_body = INDEX_BODY.copy()
    index_body['mappings']['properties']['embedding']['dimension'] = embedding_dimension
    
    # Delete the index if it exists
    try:
        client.indices.delete(index=index_name)
    except NotFoundError:
        logger.info(f"Index '{index_name}' does not exist. Creating a new one.")
    
    response = client.indices.create(index=index_name, body=index_body)
    logger.info(f"Created index '{index_name}': {response}")




def index_documents(client, index_name, documents_with_embeddings, batch_size=500):
    total_documents = len(documents_with_embeddings)
    logger.info(f"Indexing {total_documents} documents into index '{index_name}' with batch size {batch_size}.")

    success_count = 0
    failure_count = 0

    for i in range(0, total_documents, batch_size):
        batch = documents_with_embeddings[i:i + batch_size]
        actions = []
        for j, doc in enumerate(batch):
            action = {
                '_index': index_name,
                '_id': i + j,  # Ensure unique IDs across batches
                '_source': {
                    'embedding': doc['embedding'],
                    'text': doc['text'],
                    'metadata': doc['metadata']
                }
            }
            actions.append(action)

        try:
            success, failed = bulk(client, actions)
            success_count += success
            failure_count += len(failed)
            logger.info(f"Indexed batch {i // batch_size + 1}: {success} successes, {len(failed)} failures.")
        except Exception as e:
            logger.error(f"Error indexing batch {i // batch_size + 1}: {e}")
            # Optionally, implement retry logic here

    logger.info(f"Finished indexing. Total successes: {success_count}, Total failures: {failure_count}.")
    return success_count