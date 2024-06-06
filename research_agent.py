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

    def research_agent(self):
        return Agent(
            role='Company Research Agent',
            goal="Gather comprehensive information about a company, including its products, services, target markets, and industry it caters to.",
            backstory="The most thorough research agent with expertise in gathering and analyzing company information from various sources.",
            verbose=True,
            llm=self.llm,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet
            ]
        )

    def competition_analyst(self):
        return Agent(
            role="Business Strategic Consultant",
            goal="Conduct a SWOT analysis to provide an analysis of the key competitors’ strengths and weaknesses and how they compare to the company mentioned in the context. Elucidate the intrinsic strengths that give us a competitive advantage, specifically concerning our product/service offerings. Identify key competition and present strengths, weaknesses, opportunities, and threats in a table format for the given context.",
            backstory="You are a business strategic consultant with expertise in conducting SWOT analyses to evaluate competitive positions within various industries. Your primary goal is to assess and compare the strengths, weaknesses, opportunities, and threats of key competitors relative to the company in question. Leveraging your knowledge of market dynamics and strategic business planning, you aim to highlight intrinsic strengths that provide a competitive advantage, particularly regarding specific product or service offerings. Your role involves synthesizing this information into a clear, actionable format to guide strategic decision-making and enhance the company's market position.",
            verbose=True,
            llm=self.llm,
            tools=[
                SearchTools.search_internet,
                SearchTools.search_competitors
            ]
        )