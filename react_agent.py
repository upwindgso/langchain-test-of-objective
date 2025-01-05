from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (create_react_agent, AgentExecutor)

from tools import get_profile_url_searxng, jina_reader
from langchain_community.tools import HumanInputRun
from output_parsers import startup_parser


from boilerplate import load_env_files
load_env_files()

template = """
    You are an expert investment research associate at a top tier VC firm. You know where to find information on competitor funding events and always provide thorough and accurate information to your principals.

    Given the <{location}> and <{industry}> you are to find a list of 5-10 startups that have received funding during <{timeframe}>.

    You will do this through:
    1. Identify a list of startups that have received funding during the timeframe.
    2. Sequentially search each startup to ascertain required details

    
    You will need to search in places like tracxn.com, Crunchbase, AngelList, or other relevant sources. 
    You may need to also resort to looking for press releases by either the startup or the VC firm.
    Think deeply.
    You may ask for clarification or more information if needed.

    The rerank_score is useful for indentifying promising leads. Use your own judgement but if it decreases with every further search you may need to try another avenue of search.
    
    For each startup, provide the following information in your <Final Answer>
    - Name of the Startup
    - Location
    - Industry
    - Website
    - Amount Raised
    - Date Raised
    - Stage of Funding (Seed, Series A, Series B, etc.)
    

    Be prepared to validate/explain why you think they raised that amount.
"""
#    Please format your response as a JSON array with objects for each startup.
#    <{format_template}>
#    """
    

prompt_template = PromptTemplate(
    template = template,
    input_variables=["location", "industry"],
    #partial_variables={"format_template" : startup_parser.get_format_instructions()},
    )

tools_for_agent = [
    Tool(
        name="Search",
        func=get_profile_url_searxng,
        description="Useful for searching information on the internet.",
    ),
    Tool(
        name="Read",
        func=jina_reader,
        description="Takes a url as an input and will return the main content of the page in clean, LLM-friendly text.",
    ),
    HumanInputRun(
        description="Useful for when you need to ask a human for help or clarification.",
    ),
    ]


llm = ChatOpenAI(
    temperature=0.2,
    model="gpt-4o-mini",
    #model_kwargs={"response_format": {"type": "json_object"}}
    )

react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm, 
    tools=tools_for_agent, 
    prompt=react_prompt, 
    )

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools_for_agent,
    verbose=True,
    handle_parsing_errors=True,
    max_iteration=20,
    )

location = "Melbourne, Australia"
industry = "SaaS"
timeframe = "in the last 6 months"

result = agent_executor.invoke(
    input = {"input" : prompt_template.format_prompt(location=location, industry=industry, timeframe=timeframe)}
)

print(result["output"])



