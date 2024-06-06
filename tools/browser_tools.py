import json
import os

import requests
from crewai import Agent, Task
from langchain.tools import tool
from unstructured.partition.html import partition_html

class BrowserTools:

    @tool("Scrape website content")
    def scrape_website_content(website):
        """Scrape content from the given website URL."""
        url = f"https://api.scrapingant.com/v2/general?url={website}&x-api-key={os.environ['SCRAPINGANT_API_KEY']}"
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def partition_and_summarize(content, agent):
        """Partition HTML content and summarize it using an agent."""
        elements = partition_html(text=content)
        content_chunks = [str(el) for el in elements]
        chunk_size = 8000
        chunks = [content_chunks[i:i + chunk_size] for i in range(0, len(content_chunks), chunk_size)]
        summaries = []
        for chunk in chunks:
            task_description = f"Analyze and summarize the content below:\n\n{chunk}"
            task = Task(agent=agent, description=task_description)
            summaries.append(task.execute())
        return "\n\n".join(summaries)

    @tool("Scrape and summarize website")
    def scrape_and_summarize_website(website):
        """Scrape and summarize a website's content."""
        agent = Agent(
            role='Principal Researcher',
            goal='Do amazing research and summaries based on the content you are working with',
            backstory="You're a Principal Researcher at a big company and you need to do research about a given topic.",
            allow_delegation=False
        )
        content = BrowserTools.scrape_website_content(website)
        summary = BrowserTools.partition_and_summarize(content, agent)
        return summary