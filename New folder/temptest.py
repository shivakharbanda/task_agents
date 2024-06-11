from dotenv import load_dotenv
from openai import OpenAI
import json
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from googlesearch import search

load_dotenv()

model = AzureChatOpenAI(
            api_key=os.environ['AZURE_API_KEY'],
            api_version=os.environ['AZURE_API_VERSION'],
            azure_endpoint=os.environ['AZURE_ENDPOINT'],
            azure_deployment=os.environ['AZURE_DEPLOYMENT_NAME']
        )

with get_openai_callback() as cb:
    initprompt = SystemMessage(content="You are a useful assistant. The user will give you a certain query, and you must identify what information you need to answer the query. From that, generate multiple elementary queries which a search engine like google can return results for. When the user enters their input, provide no explanation and write no introduction. You must only return a each subquery on a new line and nothing else.")
    mainquery = HumanMessage(content="I want to eat some spicy pakode but they shouldn't be too oily.")
    allmessages = [initprompt, mainquery]
    print(model.invoke(allmessages))
    response = model.invoke(allmessages)
    print(response.content.split("\n"))
    print(cb.total_cost)
    results = search('abcd song', num_results=10)
    for result in results:
        print(result)