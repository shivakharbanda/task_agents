from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, WebsiteSearchTool
from textwrap import dedent
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os

# Load environment variables from .env file
load_dotenv()

class CompanyAnalysisCrew:

    def __init__(self, query) -> None:
        self.query = query
        self.llm = AzureChatOpenAI(
            api_key=os.environ['AZURE_API_KEY'],
            api_version=os.environ['AZURE_API_VERSION'],
            azure_endpoint=os.environ['AZURE_ENDPOINT'],
            azure_deployment=os.environ['AZURE_DEPLOYMENT_NAME']
        )

    def run(self):
        search_tool = SerperDevTool()
        web_rag_tool = WebsiteSearchTool()

        search_agent = Agent(
            role='Google Search Expert',
            goal='Take the user\'s natural language inputs and turn them into effective, concise Google queries.',
            backstory='You are an AI assistant trained to convert natural language questions into effective Google search queries.',
            tools=[search_tool, web_rag_tool],
            verbose=True,
            llm=self.llm,
            allow_delegation=True,
        )

        search_task = Task(
            description=dedent(f"""
                Get me the concise Google query for: {self.query}. Follow this process:
                Steps: 
 
                Identify Key Topics and Keywords: 
                Break down the input into its main components. 
                Extract nouns, verbs, and adjectives that are central to the query. 
                
                Remove Unnecessary Words: 
                Eliminate filler words (e.g., "can you," "please," "tell me about"). 
                Discard redundant phrases that do not add value to the search intent. 
                
                Prioritize Keywords: 
                Place the most important keywords at the beginning of the query. 
                Ensure that the core subject of the query is clear and prominent. 
                
                Use Concise Phrasing: 
                Convert questions into statements where applicable. 
                Use shorthand or common search phrases to streamline the query. 
                
                Add Necessary Context: 
                Include additional context if it helps clarify the search intent (e.g., specifying "next weekend" for a weather query). 
                Use location names, dates, or specific terms to narrow down the results. 
                
                Rearrange for Clarity: 
                Ensure that the query is logically structured. 
                Avoid ambiguous wording that might confuse the search engine.

                Ensure to provide only the most relevant keywords and search results.
            """),
            agent=search_agent,
            expected_output="A detailed report with the query intent, best Google query, and top search results.",
        )

        crew = Crew(
            agents=[search_agent],
            tasks=[search_task],
            verbose=True
        )

        result = crew.kickoff()
        print(result)



if __name__ == "__main__":
    print("## Welcome to Company Analysis Crew")
    user_query = input(dedent("Please specify your search query:\n"))

    query_crew = CompanyAnalysisCrew(user_query)
    query_crew.run()
