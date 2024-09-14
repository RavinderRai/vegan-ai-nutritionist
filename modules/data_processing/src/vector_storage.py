from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from opensearchpy.exceptions import NotFoundError
from opensearchpy.helpers import bulk

from config import INDEX_BODY
from ...utils.logger import setup_logger

logger = setup_logger("vector_storage", "data_processing.log")

def opensearch_client(OPENSEARCH_ENDPOINT, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION):
    awsauth = AWS4Auth(AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, 'es')
    
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



def index_documents(client, index_name, documents_with_embeddings):
    actions = []
    
    for i, doc in enumerate(documents_with_embeddings):
        action = {
            '_index': index_name,
            '_id': i,
            '_source': {
                'embedding': doc['embedding'],
                'text': doc['text'],
                'metadata': doc['metadata']
            }
        }
        actions.append(action)
    
    success, _ = bulk(client, actions)
    
    return success