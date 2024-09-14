from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ...utils.logger import setup_logger

logger = setup_logger("data_transformer", "data_processing.log")

def transform_paper_data(pdf_data: dict) -> list:
    """
    Transforms the data from one pdf file into the format we want to store it in the database. 
    In this case, the section of the paper is added to the meta data.

    Args:
        pdf_data (dict): The data of one pdf file. It contains the content of the pdf file and the original meta data.

    Returns:
        list: A list of dictionaries, where each dictionary contains the text of one section of the paper and the meta data of the paper with the section added to it.
    """

    pdf_content, original_meta_data = pdf_data['content'], pdf_data['meta_data']
    
    updated_data = []

    for text_section in pdf_content:
        meta_data = original_meta_data.copy()
        
        meta_data['section'] = text_section['section']
        
        body = text_section['body']
        
        updated_data.append({'body': body, 'meta_data': meta_data})
        
    return updated_data

def get_full_data(data):
    full_pdf_data = []

    for pdf_data in data:
        pdf_content_text = pdf_data['content']
        og_meta_data = pdf_data['meta_data']
        
        full_pdf_data += transform_paper_data(pdf_content_text, og_meta_data)
    
    return full_pdf_data


def convert_to_doc_format(transformed_data):
    documents = []
    for content in transformed_data:
        doc = Document(page_content=content['body'], metadata=content['meta_data'])
        documents.append(doc)
    return documents

def chunk_text(document, chunk_size=10000, chunk_overlap=1000):
    raw_text = document.page_content
    meta_data = document.metadata
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    texts = text_splitter.split_text(raw_text)
    docs = [Document(page_content=t, metadata=meta_data) for t in texts]
    
    return docs

def chunk_doc(document, chunk_size=10000, chunk_overlap=1000):
    chunked_docs = []
    
    for doc in document:
        chunked_docs += chunk_text(doc, chunk_size, chunk_overlap)
    
    return chunked_docs