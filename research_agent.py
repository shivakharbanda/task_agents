from crewai import Agent

from tools.browser_tools import BrowserTools
from tools.search_tools import SearchTools
from langchain_openai import  AzureChatOpenAI

deployment_name='gpt-35-turbo'

llm = AzureChatOpenAI(
        api_key="bd18995c51fa40e19e493df21c7ded81",  
        api_version="2024-02-01",
        azure_endpoint = "https://madhukar-kumar.openai.azure.com/",
        azure_deployment=deployment_name
    )
class TaskAgents:
    def research_agent(self):
        return Agent(
            role='Company Research Agent',
            goal="""Gather comprehensive information about a company, including 
            its products, services, target markets, and industry it caters to.""",
            backstory="""The most thorough research agent with expertise in gathering 
            and analyzing company information from various sources.""",
            verbose=True,
            llm = llm,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet
            ]
        )

    def competition_analyst(self):
        return Agent(
            role='Competition Analyst',
            goal="""Analyze competitors of a company based on the information collected 
            by the research agent. Compare and rank competitors, and generate a report.""",
            backstory="""A highly skilled competition analyst known for their ability 
            to evaluate competitors and provide insightful comparisons and rankings.""",
            verbose=True,
            llm = llm,
            tools=[
                SearchTools.search_internet,
                SearchTools.search_competitors
            ]
        )
