from crewai import Task
from textwrap import dedent

class CompanyAnalysisTasks:
    def research(self, agent, company, website, industry):
        return Task(
            description=dedent(f"""
                Gather comprehensive information about the company. 
                This includes its products, services, target markets, 
                and the industry it caters to. Also look for recent news 
                and significant events.

                Your final report MUST include:
                - Detailed descriptions of the company's products and services.
                - Identification of the target markets.
                - Summary of the industry the company belongs to.
                - Any recent news or significant events affecting the company.

                Ensure to use the most recent data available.

                Selected company by the customer: {company}
                Company website: {website}
                Industry: {industry}
            """),
            agent=agent,
            expected_output="A detailed report with descriptions of the company's products, services, target markets, industry summary, and recent significant events."
        )

    def competition_analysis(self, agent, company_info):
        return Task(
            description=dedent(f"""
                Analyze competitors of the company based on the information collected 
                by the research agent. Identify companies operating in the same market 
                and industry. Compare and rank these competitors against the main company.

                Your final report MUST include:
                - Names and brief descriptions of key competitors.
                - Comparison and ranking of competitors against the main company.
                - Insights on the competitive landscape and market position.

                Make sure to provide a comprehensive and detailed analysis.

                Information provided by the research agent: {company_info}
            """),
            agent=agent,
            expected_output="A comprehensive report with names, descriptions, comparisons, and rankings of key competitors, along with insights on the competitive landscape."
        )
