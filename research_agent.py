import os

from crewai import Agent

from tools.browser_tools import BrowserTools
from tools.search_tools import SearchTools
from langchain_openai import  AzureChatOpenAI
class TaskAgents:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            api_key=os.environ['AZURE_API_KEY'],
            api_version=os.environ['AZURE_API_VERSION'],
            azure_endpoint=os.environ['AZURE_ENDPOINT'],
            azure_deployment=os.environ['AZURE_DEPLOYMENT_NAME']
        )

    def planner_agent(self):
        return Agent(
            role='Planner',
            goal="Analyze the provided problem description and generate a comprehensive plan, identifying all necessary tasks and determining their order and dependencies.",
            backstory="You are a strategic planner with expertise in breaking down complex problems into manageable tasks and creating detailed plans for their execution.",
            verbose=True,
            llm=self.llm,
        )

    def executor_agent(self):
        return Agent(
            role='Executor',
            goal="Execute the given tasks using the best available tools, and if a suitable tool is not found, identify and describe the required tool and raise an exception.",
            backstory="You are a task executor with expertise in using various tools to complete tasks efficiently and effectively. You cannot delegate tasks further.",
            verbose=True,
            llm=self.llm,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet,
                # Add more tools as needed
            ]
        )