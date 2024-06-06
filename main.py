from crewai import Crew
from textwrap import dedent
from dotenv import load_dotenv

from research_agent import TaskAgents
from research_tasks import CompanyAnalysisTasks

# Load environment variables from .env file
load_dotenv()

class CompanyAnalysisCrew:
    def __init__(self, company, website, industry):
        self.company = company
        self.website = website
        self.industry = industry

    def run(self):
        agents = TaskAgents()
        tasks = CompanyAnalysisTasks()

        research_agent = agents.research_agent()
        competition_analyst_agent = agents.competition_analyst()

        research_task = tasks.research(research_agent, self.company, self.website, self.industry)
        competition_task = tasks.competition_analysis(competition_analyst_agent, research_task)

        crew = Crew(
            agents=[
                research_agent,
                competition_analyst_agent
            ],
            tasks=[
                research_task,
                competition_task
            ],
            verbose=True
        )

        result = crew.kickoff()
        return result

if __name__ == "__main__":
    print("## Welcome to Company Analysis Crew")
    print('-------------------------------')
    company = input(
        dedent("""
            What is the name of the company you want to analyze?
        """)
    )
    website = input(
        dedent("""
            What is the website of the company?
        """)
    )
    industry = input(
        dedent("""
            What industry does the company belong to?
        """)
    )

    company_analysis_crew = CompanyAnalysisCrew(company, website, industry)
    result = company_analysis_crew.run()
    print("\n\n########################")
    print("## Here is the Report")
    print("########################\n")
    print(result)