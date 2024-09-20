import boto3
from langchain_aws import BedrockEmbeddings

import logging

logger = logging.getLogger(__name__)

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
    logger.info(f"Initializing BedrockEmbeddings with model_id: {model_id}")
    bedrock = boto3.client(service_name="bedrock-runtime")
    
    bedrock_embeddings = BedrockEmbeddings(
        model_id=model_id,
        client=bedrock
    )
    
    return bedrock_embeddings


def generate_bedrock_embeddings(documents, bedrock_embeddings):
    """
    Generate embeddings for a list of documents and return a new list of documents 
    with their respective embeddings, computed using the Bedrock embeddings service and titan model.

    Args:
        documents (list[Document]): The list of documents to generate embeddings for.
        bedrock_embeddings (BedrockEmbeddings): The client that can interact with the Bedrock embeddings service.

    Returns:
        list[dict]: A list of documents where each document has the embedding included as a key.
    """
    logger.info("Generating embeddings for documents...")
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
    
    logger.info("Embeddings generated successfully.")
    return documents_with_embeddings