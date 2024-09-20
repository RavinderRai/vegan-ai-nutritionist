from typing import List, Callable
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

def transform_paper_data(pdf_data: dict) -> list:
    """
    Transforms the data from one pdf file into the format we want to store it in the database. 
    In this case, the section of the paper is added to the meta data.

    Args:
        pdf_data (dict): The data of one pdf file. It contains the content of the pdf file and the original meta data.

    Returns:
        list: A list of dictionaries, where each dictionary contains the text of one section of the paper and the meta data of the paper with the section added to it.
    """
    logger.info("Transforming pdf data, moving section to meta data.")
    pdf_content, original_meta_data = pdf_data['content'], pdf_data['meta_data']
    
    updated_data = []

    for text_section in pdf_content:
        meta_data = original_meta_data.copy()
        
        meta_data['section'] = text_section['section']
        
        body = text_section['body']
        
        updated_data.append({'body': body, 'meta_data': meta_data})
        
    logger.info("Data transformed successfully.")
    return updated_data

def get_full_data(data):
    logger.info("Aggregating full data from all PDFs...")
    full_pdf_data = []

    for pdf_data in data:
        full_pdf_data += transform_paper_data(pdf_data)
    
    logger.info("Full data aggregation complete.")
    return full_pdf_data


def convert_to_doc_format(transformed_data):
    logger.info("Converting transformed data into Langchain Document format.")
    documents = []
    for content in transformed_data:
        doc = Document(page_content=content['body'], metadata=content['meta_data'])
        documents.append(doc)
    
    logger.info("Conversion to Document format complete.")
    return documents

def chunk_text(document, chunk_size=5000, chunk_overlap=500):
    """
    Chunk text into smaller pieces based on a given chunk size and overlap.
    Chunks by text/string, not tokens.

    Args:
        document (Document): The document to be chunked.
        chunk_size (int): The size of each chunk in characters. Defaults to 5000.
        chunk_overlap (int): The overlap between each chunk in characters. Defaults to 500.

    Returns:
        List[Document]: The list of documents created by chunking the input document.
    """

    raw_text = document.page_content
    meta_data = document.metadata
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    texts = text_splitter.split_text(raw_text)
    docs = [Document(page_content=t, metadata=meta_data) for t in texts]
    
    return docs


def chunk_doc(documents: List[Document], chunk_size: int = 10000, chunk_overlap: int = 1000) -> List[Document]:
    """
    Chunk a list of documents into smaller documents based on a given chunk size and overlap.
    Uses the chunk_text funtion, meant to chunk the full dataset.

    Args:
        documents (List[Document]): The list of documents to be chunked.
        chunk_size (int): The size of each chunk in characters. Defaults to 10000.
        chunk_overlap (int): The overlap between each chunk in characters. Defaults to 1000.

    Returns:
        List[Document]: The chunked documents.
    """
    
    logger.info("Chunking documents...")
    chunked_docs: List[Document] = []
    
    for doc in documents:
        chunked_docs += chunk_text(doc, chunk_size, chunk_overlap)
    
    return chunked_docs



def chunk_documents_by_tokens(
    documents: List[Document], 
    model_name: str, 
    chunk_size: int = 2048, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Chunk a list of documents by their token count using a given model's tokenizer.

    Args:
        documents (List[Document]): The list of documents to be chunked.
        model_name (str): The name of the model to use for tokenization.
        chunk_size (int): The size of each chunk in tokens. Defaults to 2048.
        chunk_overlap (int): The overlap between each chunk in tokens. Defaults to 200.

    Returns:
        List[Document]: The chunked documents.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tiktoken_len(text: str) -> int:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)
