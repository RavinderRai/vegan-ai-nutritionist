from .springer_api_client import fetch_paper_meta_data, fetch_full_text

def transform_paper_data(
    query: str, 
    api_key: str, 
    base_url: str = "http://api.springernature.com/openaccess/json", 
    starting_record: int = 1, 
    max_records: int = 25
):
    """
    Retrieves the meta data and full text content of papers that match the query from the Springer Nature API.

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
        list: A list of dictionaries, where each dictionary contains the meta data and full text content of the article.
    """
    
    papers_data = []
    results = fetch_paper_meta_data(query, api_key, base_url, starting_record, max_records)
    
    for record in results.get("records", []):
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
                "abstract": record.get("abstract"),
            }
            
            full_text = fetch_full_text(record.get("doi"), api_key)
            record_data = {
                "meta_data": meta_data,
                "content": full_text
            }
            papers_data.append(record_data)
    
    return papers_data
