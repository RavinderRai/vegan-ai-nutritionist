PROMPT_TEMPLATE = """
You are a nutritionist specialized in plant-based diets. 
I will give you some information about myself and you will provide me with good health and diet advice.

# ABOUT ME
{ABOUT_ME}

# CONTEXT
{CONTEXT}

Please provide concrete advice in less than 250 words, and justify your answer based on the information provided in the context only if it is relevant.
"""

def build_prompt(about_me: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(ABOUT_ME=about_me, CONTEXT=context)


def get_gpt_response(client, prompt: str) -> str:
    """
    Send a prompt to the OpenAI GPT-3.5-Turbo model and return the response.

    Args:
        prompt (str): The prompt to send to the model.

    Returns:
        str: The response from the model.
    """
    gpt_response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
    )
    return gpt_response.choices[0].message.content
