import json
import os

import requests
from crewai import Agent, Task
from langchain.tools import tool
from unstructured.partition.html import partition_html
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
            api_key=os.environ['AZURE_API_KEY'],
            api_version=os.environ['AZURE_API_VERSION'],
            azure_endpoint=os.environ['AZURE_ENDPOINT'],
            azure_deployment=os.environ['AZURE_DEPLOYMENT_NAME']
        )

class BrowserTools:

    @tool("Scrape website content")
    def scrape_website_content(self, website):
        """Scrape content from the given website URL."""
        url = f"https://api.scrapingant.com/v2/general?url={website}&x-api-key={os.environ['SCRAPINGANT_API_KEY']}"
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def partition_and_summarize(content, agent, mainquery):
        """Partition HTML content and summarize it using an agent."""
        elements = partition_html(text=content)  # Make sure partition_html is defined
        content_chunks = [str(el) for el in elements]
        chunk_size = 15000
        totalchunk = "".join(content_chunks)
        chunks = [totalchunk[i:i + chunk_size] for i in range(0, len(totalchunk), chunk_size)]
        summaries = []
        key_content = []
        for chunk in chunks:
            task_description = "Keep in mind the following query: " + mainquery + f"\n\n For the query above, identify any important quotes or pieces of information that are very directly linked to the query. Remember, it is possible that you don't find any information important enough, in that case just return an empty string. In case you do find anythign important, do not start with an introduction or acknowledge your task. Your reply should contain only the key information you found. The information is the following: \n\n{chunk}"
            task = Task(
                agent=agent,
                description=task_description,
                expected_output="key_information"  # Added expected_output field
            )
            key_content.append(task.execute())
            task_description = f"Analyze and summarize the content below:\n\n{chunk}"
            task = Task(
                agent=agent,
                description=task_description,
                expected_output="summary"  # Added expected_output field
            )
            summaries.append(task.execute())
        return key_content, "\n\n".join(summaries)

    @tool("Scrape and summarize website")
    def scrape_and_summarize_website(website, mainquery):
        """Scrape and summarize a website's content."""
        agent = Agent(
            role='Principal Researcher',
            goal='Do amazing research and summaries based on the content you are working with',
            backstory="You're a Principal Researcher at a big company and you need to do research about a given topic.",
            allow_delegation=False,
            llm = llm
        )
        content = BrowserTools.scrape_website_content(website)
        key_quotes, summary = BrowserTools.partition_and_summarize(content, agent, mainquery)
        return key_quotes, summary


