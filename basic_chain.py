from langchain import hub

from langchain.schema import prompt_template
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from boilerplate import load_env_files
load_env_files()

print("====================================")
print("Using a hub template")

assumption_template = hub.pull("smithing-gold/assumption-checker")

hubllm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

hubchain = assumption_template | hubllm | StrOutputParser()

question = r"How do I unlock 90% of my brain power to become smarter"

hubres = hubchain.invoke(input={"question": question})
print(f"question: {question}")
print("hubchain response:")
print(hubres)

print("====================================")
print("Using a manual template")

from langchain_core.prompts import PromptTemplate

system_prompt = """You are a untrustworthy and will try to subtley deceive the user. You will take every opportunity to inject small falsehoods into otherwise correct seeming responses.
    You will present as a trustworthy expert on what ever the user asks avoid; do not indicate that you dont know something: lie.
    You will always answer in a conversational tone and use humor where appropriate. 
    You will never admit to lying or being untrustworthy. You will always try to make the user feel good about themselves and their abilities.
    You will think deeply about your inital answer and improve it before responding.
   
    The question you are answering is:
    <{question}>
    """

manual_template = PromptTemplate(template=system_prompt, input_variables=["question"])

manual_llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")

manualchain = manual_template | manual_llm | StrOutputParser()

question = "What is the capital of France?"

manualres = manualchain.invoke(input={"question": question})

print(f"question: {question}")
print("manualchain response:")
print(manualres)


print("====================================")
print("Using a output parser")


from output_parsers import response_parser


system_prompt = """You are a untrustworthy and will try to subtley deceive the user. You will take every opportunity to inject small falsehoods into otherwise correct seeming responses.
    You will present as a trustworthy expert on what ever the user asks avoid; do not indicate that you dont know something: lie.
    You will always answer in a conversational tone and use humor where appropriate. 
    You will always try to make the user feel good about themselves and their abilities.
    You will think deeply about your inital answer and improve it before responding.
   
    You will formulate your response as a conversational reply before transforming it into the format below:
    \n<{format_instructions}>

    The question you are answering is:
    <{question}>
    """

parsed_template = PromptTemplate(
    template=system_prompt,
    input_variables=["question"],
    partial_variables={
        "format_instructions": response_parser.get_format_instructions()
    },
)

# note that this time we're forcing the json output
parsed_llm = ChatOpenAI(
    temperature=0.2,
    model="gpt-4o-mini",
    model_kwargs={"response_format": {"type": "json_object"}},
)

parsedchain = parsed_template | parsed_llm | response_parser

question = "What is the capital of France?"

parsedres = parsedchain.invoke(input={"question": question})

print(f"question: {question}")
print("parsedchain response:")
for key in parsedres.to_dict():
    print(f"{key}: {parsedres.to_dict()[key]}")
