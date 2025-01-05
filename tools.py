from langsmith.run_trees import LANGSMITH_PROJECT
from boilerplate import load_env_files
load_env_files()

import requests
import os

import re
from urllib.parse import unquote

import re
from langchain.schema import BaseOutputParser
from langchain import hub
from langchain_openai import ChatOpenAI

from langsmith import traceable

class FactCheckOutput():
    def __init__(self, parsed_output):
       self.assumption = parsed_output['assumption']
       self.followup = parsed_output['followup']

class FactCheckParser(BaseOutputParser):
    """Parse the output of an LLM call """

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        assumption_matches = re.findall(r'Assumption: (.+)', text)
        followup_matches = re.findall(r'Fact Check: (.+)', text)

        return FactCheckOutput({
            'assumption': assumption_matches if len(assumption_matches) > 0 else None,
            'followup': followup_matches if len(followup_matches) > 0 else None
      })

@traceable(project_name="assumption_checker")    #set the project_name by function
def assumption_checker(question:str):
    """Checks an assumption using a fact checker API. Returns an array of dicts with the paired assumption and follow-up question required to fact check the assumption."""
    
    assumption_template = hub.pull("smithing-gold/assumption-checker")


    llm = ChatOpenAI(
        temperature=0.1, 
        model="gpt-4o-mini", 
        )

    chain = assumption_template | llm | FactCheckParser()

    question = r"How do I unlock 90% of my brain power to become smarter"

    res = chain.invoke(input={"question": question})
    print(f"question: {question}")
    print(f"assumption: {res.assumption}")
    print(f"followup:  {res.followup}")
    
    output = []
    for i in range(len(res.assumption)):
        output.append({
            "assumption": res.assumption[i],
            "followup": res.followup[i]
        })

    return output


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
    
    print(assumption_checker("How can I unlock the unused 90% of my brain to become smarter?"))
    #print(jina_reader("https://www.whoraised.io/saas-startups/melbourne-saas-startups"))
