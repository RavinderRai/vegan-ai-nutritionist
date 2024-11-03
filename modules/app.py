import os
import boto3
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain_aws import BedrockEmbeddings

from inference.src.model_inference import ModelInference


INDEX_NAME = "vegan_papers_index"

PROMPT_TEMPLATE = """
You are a Vegan Nutritionist specialized in vegan and plant-based nutrition.
Answer the question based on your expert knowledge in a clear and concise manner, with around 250 words.
Below you are provided with a few snippets of research papers related to the question, along with the title and link of the paper they belong to.
Please use the provided research paper excerpts to support your answer, but only use the relevant ones.

And absolutely make sure to include the link to the papers you reference, but if and only if you used it to support your answer! 
Again, please do not forget to include any links to papers you reference in your answer.

Contexts: {contexts}

Question: {question}
"""

load_dotenv(find_dotenv())

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION")

OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT")

bedrock_client = boto3.client(service_name="bedrock-runtime")

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, 
    input_variables=["contexts", "question"]
)

def get_llm(client, light_model=False):
    """
    Returns a Bedrock Large Language Model (LLM) for generating text based on prompts.

    Args:
        client (boto3.client): The AWS client to use for calling the Bedrock API.
        light_model (bool, optional): Whether to use the small 8B model or the larger 70B model. Defaults to False.

    Returns:
        Bedrock: The LLM object that can be used to generate text based on prompts.
    """
    if light_model:
        model_id = "meta.llama3-8b-instruct-v1:0"
    else:
        model_id = "meta.llama3-70b-instruct-v1:0"
        
    # meta.llama3-8b-instruct-v1:0
    llm=Bedrock(
        model_id=model_id,
        client=client,
        model_kwargs={'max_gen_len':512}
    )
    return llm

def embed_query(query_text, client):
    """
    Embeds a query text into a vector using the Bedrock text embeddings service.

    Args:
        query_text (str): The text to embed.
        client (boto3.client): The AWS client to use for calling the Bedrock API.

    Returns:
        list[float]: The embedded vector.
    """
    bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=client)
    query_embedding = bedrock_embeddings.embed_query(query_text)
    
    # new error, seems to be a mismatch where 
    # the query embedding is 1536 but size 384 is expected
    # truncating the query embedding to 384 for now
    return query_embedding[:384]


def similarity_search(
    query_embedding: List[float], 
    index_name: str, 
    AWS_ACCESS_KEY: str, 
    AWS_SECRET_KEY: str, 
    AWS_REGION: str, 
    OPENSEARCH_ENDPOINT: str, 
    top_k: int = 3
) -> List[Dict[str, str]]:
    """
    Perform a similarity search in the OpenSearch index using the given query embedding.

    Parameters:
        query_embedding (List[float]): The embedding vector of the query.
        index_name (str): The name of the OpenSearch index.
        AWS_ACCESS_KEY (str): The AWS access key ID.
        AWS_SECRET_KEY (str): The AWS secret access key.
        AWS_REGION (str): The AWS region.
        OPENSEARCH_ENDPOINT (str): The OpenSearch endpoint.
        top_k (int, optional): The number of similar documents to return. Defaults to 3.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the title, link, and text of the similar documents.
    """
    awsauth = AWS4Auth(AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, 'es')

    client = OpenSearch(
        hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    
    search_body = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_k
                }
            }
        },
        "_source": ["text", "metadata"]
    }

    response = client.search(index=index_name, body=search_body)
    
    hits = []
    
    for hit in response['hits']['hits']:
        hit_dct = {
            'title': hit['_source']['metadata'].get('title', 'No Title'),
            'link': hit['_source']['metadata']['url'][0].get('value', 'No Link'),
            'text': hit['_source']['text']
        }
        hits.append(hit_dct)
        
    return hits

def main(index_name):
    """
    Main entry point of the Streamlit app.

    Parameters
    ----------
    index_name : str
        The name of the OpenSearch index to search in.

    """
    st.set_page_config(page_title="AI Vegan Nutritionist")
    st.header("Chat with a Vegan AI Nutritionist.")
    st.subheader("Powered by the latest research at Springer Nature!")
    
    model_choice = st.sidebar.radio(
        "Choose a model to chat with:",
        ("Bedrock Llama LLM", "Fine-tuned Falcon-7B-Instruct LLM")
    )
    
    user_query = st.text_input("Ask a question about vegan or plant-based diets:")
    
    if st.button("Submit"):
        with st.spinner("Generating Answer..."):
            query_embedding = embed_query(user_query, bedrock_client)
            
            search_results = similarity_search(query_embedding, index_name, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, OPENSEARCH_ENDPOINT)

            # Prepare contexts for the prompt
            contexts = ""
            for idx, result in enumerate(search_results, start=1):
                title = result['title']
                link = result['link']
                text = result['text']
                contexts += f"\n<Paper {idx} Title>\n{title}\n<Paper {idx} Title>\n"
                contexts += f"\n<Paper {idx} Link>\n{link}\n<Paper {idx} Link>\n"
                contexts += f"\n<Paper {idx} Context>\n{text}\n<Paper {idx} Context>\n"
        
            prompt = PROMPT_TEMPLATE.format(contexts=contexts, question=user_query)

            
            if model_choice == "Bedrock Llama LLM":
                llm = get_llm(bedrock_client)
                response = llm(prompt)
            else:
                model_inference = ModelInference()
                response = model_inference.predict(prompt)
            
            st.write(response)
        
if __name__ == "__main__":
    main(INDEX_NAME)