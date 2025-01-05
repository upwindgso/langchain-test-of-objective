from boilerplate import load_env_files
load_env_files()

import requests
import os

import re
from urllib.parse import unquote

def clean_url(url):
    decoded_url = unquote(url)
    cleaned_url = re.sub(r'["<>#%{}|\\^~\[\]`]', '', decoded_url)

    cleaned_url = cleaned_url.replace('Read the link "', '').strip()

    return cleaned_url

def rerank_results(query, passages):
    """Reranks the results based on relevance to the name."""
    response = requests.post(
        "http://localhost:8000/rerank",
        json={"query": query, "passages": passages, "use_llm": False}
    )
    return response.json()["response"]

def jina_reader(url: str):
    """This will return the main content of the page in clean, LLM-friendly text.."""

    #print("============ jina reader")
    #print(url)

    url = 'https://r.jina.ai/'+clean_url(url)
    
    print(url)
    headers = {
        'Authorization': 'Bearer ' + os.getenv('JINA_API_KEY') 
    }

    response = requests.get(url, headers=headers)

    return response.text

from langchain_community.utilities import SearxSearchWrapper

def get_profile_url_searxng(name : str, location: str = "", keywords: str = ""):
    """Searches for Linkedin or Twitter Profile page"""

    search = SearxSearchWrapper(
        searx_host="http://localhost:8080"
    )
    res = search.results(
        f"{name}",
        num_results=10,
        
        )
    
    # Extract snippets for reranking
    content = [f"{item.get('title', '')} - {item.get('snippet', '')}" for item in res]

    query = str(
            {"name" : name,
             "location" : location,
             "keywords" : keywords
             }
    )
    
    # Get reranking scores
    scores = rerank_results(query, content)

    # Add scores back to original results
    for item, score in zip(res, scores):
        item['rerank_score'] = float(score)  # Convert tensor to float if needed

    # Optional: Sort results by rerank score
    res = sorted(res, key=lambda x: x['rerank_score'], reverse=True)
    
    return res

if __name__ == "__main__":
    
    print(jina_reader("https://www.whoraised.io/saas-startups/melbourne-saas-startups"))
