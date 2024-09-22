import json
from tqdm import tqdm
from .examples import EXAMPLES
from .api_clients import get_openai_client, get_opensearch_client
from .retrieval import get_query_embedding, get_context
from .gpt import build_prompt, get_gpt_response
from .config import DATA_PATH
from ...utils.logger import setup_logger

def run():
    full_q_and_a_data = []
    
    logger = setup_logger("generating_training_data", "generating_training_data.log")
    logger.info("Starting to iteratively generate training data.")
    
    logger.info("Setting up OpenAI API client...")
    gpt_client = get_openai_client()
    
    logger.info("Setting up OpenSearch client...")
    opensearch_client = get_opensearch_client()
    
    
    
    for example in tqdm(EXAMPLES):
        query_text = example['about_me'] + " " + example['question']
        query_embedding = get_query_embedding(query_text)
        context = get_context(opensearch_client, query_embedding)
        prompt = build_prompt(query_text, context)
        response = get_gpt_response(gpt_client, prompt)

        full_q_and_a_data.append({
            **example, 
            "context": context, 
            "response": response
        })
        
    logger.info("Training data generated successfully.")
    
    with open(DATA_PATH / "training_data.json", "w") as f:
        json.dump(full_q_and_a_data, f, indent=4)
        
    logger.info("Training data saved successfully.")

if __name__ == "__main__":
    run()
