import boto3
from langchain_aws import BedrockEmbeddings

from ...utils.logger import setup_logger

logger = setup_logger("embeddings", "data_processing.log")

def get_bedrock_embeddings(model_id: str = "amazon.titan-embed-text-v1"):
    """
    Get a BedrockEmbeddings object which is a client that can interact with the
    Bedrock embeddings service.

    Args:
        model_id (str, optional): The identifier of the model to use. Defaults to
            "amazon.titan-embed-text-v1".

    Returns:
        BedrockEmbeddings: A client that can interact with the Bedrock embeddings
            service.
    """

    bedrock = boto3.client(service_name="bedrock-runtime")
    
    bedrock_embeddings = BedrockEmbeddings(
        model_id=model_id,
        client=bedrock
    )
    
    return bedrock_embeddings


def generate_embeddings(documents, bedrock_embeddings):
    """
    Generate embeddings for a list of documents and return a new list of documents with embeddings included.

    Args:
        documents (list[Document]): The list of documents to generate embeddings for.
        bedrock_embeddings (BedrockEmbeddings): The client that can interact with the Bedrock embeddings service.

    Returns:
        list[dict]: A list of documents where each document has the embedding included as a key.
    """
    texts = [doc.page_content for doc in documents]
    
    embeddings = bedrock_embeddings.embed_documents(texts)
    
    documents_with_embeddings = []
    
    for doc, embedding in zip(documents, embeddings):
        doc_with_embedding = {
            "embedding": embedding,
            "text": doc.page_content,
            "metadata": doc.metadata
        }
        documents_with_embeddings.append(doc_with_embedding)
    
    return documents_with_embeddings