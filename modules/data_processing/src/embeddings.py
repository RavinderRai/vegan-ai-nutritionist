from sentence_transformers import SentenceTransformer
from typing import List, Dict
from langchain.docstore.document import Document

import logging

logger = logging.getLogger(__name__)

def get_embedding_model(model_id: str) -> SentenceTransformer:
    """
    Returns a SentenceTransformer model with the given model_id.

    Args:
        model_id (str): The id of the model to use.

    Returns:
        SentenceTransformer: The SentenceTransformer model.
    """

    logger.info(f"Initializing Embeddings with model_id: {model_id}")
    return SentenceTransformer(model_id)


def generate_embeddings(documents: List[Document], embedding_model: SentenceTransformer):
    """
    Generate embeddings for a list of documents and return a new list of documents 
    with their respective embeddings, computed using an open source sentence transformer model.

    Args:
        documents (List[Document]): The list of documents to generate embeddings for.
        embedding_model (SentenceTransformer): The open source sentence transformer model to use.

    Returns:
        List[Dict[str, Union[List[float], str, Dict[str, str]]]]: A list of documents where each document has the embedding included as a key.
    """
    logger.info("Generating embeddings for documents...")
    texts = [doc.page_content for doc in documents]
    
    embeddings = embedding_model.encode(texts)
    
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
