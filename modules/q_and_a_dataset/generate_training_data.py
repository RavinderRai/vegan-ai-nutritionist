import os
import openai
from tqdm import tqdm
from typing import Dict
from dotenv import load_dotenv, find_dotenv

from .examples import EXAMPLES
from .config import DATA_DIR

load_dotenv(find_dotenv())

openai.api_key = os.environ.get("OPENAI_API_KEY")

PROMPT_TEMPLATE = """
You are a nutritionist specialized in plant-based diets. 
I will give you some information about myself and you will provide me with good health and diet advice.

# ABOUT ME
{ABOUT_ME}

# CONTEXT
{CONTEXT}

Please provide concrete advice in less than 250 words, and justify your answer based on the information provided in the context.
"""

def build_prompt(example: Dict) -> str:
    return PROMPT_TEMPLATE.format(
        ABOUT_ME=example["about_me"],
        CONTEXT=example["context"],
    )

def run():
    output = []
    for example in tqdm(EXAMPLES):
        prompt = build_prompt(example)

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=5,
            max_tokens=100,
        )

        response = response["choices"][0]["text"]

        output.append({**example, "response": response})

    # save output as json file
    import json

    with open(DATA_DIR / "training_data.json", "w") as f:
        json.dump(output, f, indent=4)