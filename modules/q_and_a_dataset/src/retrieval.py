from sentence_transformers import SentenceTransformer
from .config import INDEX_NAME, EMBEDDING_MODEL_ID

def get_query_embedding(query_text: str, EMBEDDING_MODEL_ID: str = EMBEDDING_MODEL_ID) -> list:
    model = SentenceTransformer(EMBEDDING_MODEL_ID)
    return model.encode(query_text)

def get_context(client, query_embedding: list, size: int = 3, index_name: str = INDEX_NAME) -> str:    
    search_body = {
        "size": size,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": 5
                }
            }
        },
        "_source": ["text", "metadata"]
    }

    response = client.search(index=index_name, body=search_body)
    
    context = " ".join([hit['_source']['text'] for hit in response['hits']['hits'][:size]])
    
    return context


