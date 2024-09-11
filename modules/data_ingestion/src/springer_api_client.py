import requests
import xml.etree.ElementTree as ET

def fetch_paper_meta_data(
    query: str, 
    api_key: str, 
    base_url: str = "http://api.springernature.com/openaccess/json", 
    starting_record: int = 1, 
    max_records: int = 25
): 
    
    """
    Retrieves the meta data for papers that match the query from the Springer Nature API.

    Args:
        query (str): Description of what kind of papers you want to collect.
        api_key (str): Springer Nature API Key. Note: there is a free version and a premium version.
        base_url (str, optional): The endpoint of the Springer Nature API to get meta data. 
            Defaults to "http://api.springernature.com/openaccess/json".
        starting_record (int, optional): The starting record number. Defaults to 1. 
            Will be set to 1 if input is lower.
        max_records (int, optional): The max number of records you can query at a time. 
            Defaults to 25, and will be set to 25 if input is higher (or lower than 1).

    Returns:
        dict: The response from the API given as a JSON object.
    """
    if starting_record < 1:
        starting_record = 1
    if not 1 <= max_records <= 25:
        max_records = 25
    
    params = {
        "q": query,
        "api_key": api_key,
        "s": starting_record,
        "p": max_records,
    }
    
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    return response.json()


def fetch_full_text(
    doi: str, 
    api_key: str, 
    base_url: str ="http://api.springernature.com/openaccess/jats"
):
    """
    Retrieves the full text content of a journal article given its DOI and API key.

    Args:
        doi (str): The DOI of the article.
        api_key (str): The API key for accessing the Springer Nature API.
        base_url (str, optional): The base URL for the API. Defaults to "http://api.springernature.com/openaccess/jats".

    Returns:
        list: A list of dictionaries, where each dictionary contains the section title and body text of the article.
        If the request fails, returns an error message.
    """
    
    params = {
        "q": doi,
        "api_key": api_key
    }
    
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    xml_content = response.content
    root = ET.fromstring(xml_content)

    full_text = []
    body_section = root.find(".//body")
    
    if body_section is not None:
        for section in body_section.findall(".//sec"):
            section_title = section.find("title")
            section_title_text = section_title.text if section_title is not None else ""
            
            paragraph_text = ""
            for paragraph in section.findall(".//p"):
                if paragraph.text:
                    paragraph_text += paragraph.text
            
            full_text.append(
                {
                    "section": section_title_text, 
                    "body": paragraph_text
                    }
            )
    
    return full_text
