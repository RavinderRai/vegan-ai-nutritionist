import requests
import xml.etree.ElementTree as ET

def get_full_text(
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
    status_code = response.status_code
    
    if status_code == 200:
        xml_content = response.content
        root = ET.fromstring(xml_content)
        
        body_section = root.find(".//body")
        full_text = []
        
        if body_section is not None:
            for section in body_section.findall(".//sec"):
                section_title = section.find("title")
                
                if section_title is not None and section_title.text:
                    section_title_text = section_title.text
                else:
                    section_title_text = ""
                
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
    else:
        return "Error: request failed with status code {}".format(status_code)


def get_paper_data(
    query: str, 
    api_key: str, 
    base_url: str = "http://api.springernature.com/openaccess/json", 
    starting_record: int = 1, 
    max_records: int = 25
): 
    """This function collects meta data for papers that match the query from the Springer Nature API.

    Args:
        query (str): Description of what kind of papers you want to collect.
        api_key (str): Springer Nature API Key. Note: there is a free version and a premium version.
        base_url (str, optional): The endpoint of the Springer Nature API to get meta data. 
            Defaults to "http://api.springernature.com/openaccess/json".
        starting_record (int, optional): The starting record number. Defaults to 1.
        max_records (int, optional): The max number of records you can query at a time. 
            Defaults to 25, and floors to 25 if set higher.

    Returns:
        list: A list of meta data dictionaries for papers that match the query.
    """
    if starting_record < 1:
        raise ValueError("starting_record must be greater than or equal to 1")
    if not 1 <= max_records <= 25:
        raise ValueError("max_records must be between 1 and 25 (inclusive)")
    
    params = {
        "q": query,
        "api_key": api_key,
        "s": starting_record,
        "p": max_records,
    }
    
    response = requests.get(base_url, params=params)
    status_code = response.status_code
    
    if status_code == 200:
        results = response.json()
        
        papers_data = []

        for record in results.get("records", []):
            # Double check if the record is open access
            if record.get("openAccess"):
                meta_data = {
                    "content_type": record.get("contentType"),
                    "url": record.get("url"),
                    "title": record.get("title"),
                    "publication_name": record.get("publicationName"),
                    "doi": record.get("doi"),
                    "publication_date": record.get("publicationDate"),
                    "starting_page": record.get("startingPage"),
                    "ending_page": record.get("endingPage"),
                    "open_access": record.get("openAccess"),
                    "abstract": record.get("abstract")
                }
                
            full_text = get_full_text(record.get("doi"), api_key)
            
            record_data = {
                "meta_data": meta_data,
                "content": full_text
            }
            
            papers_data.append(record_data)
            
        return papers_data
    else:
        return "Error: request failed with status code {}".format(status_code)
    