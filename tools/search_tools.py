import json
import os

import requests
from langchain.tools import tool

class SearchTools:
    
    @tool("Search the internet")
    def search_internet(query):
        """Search the internet and return relevant results."""
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {'X-API-KEY': os.environ['SERPER_API_KEY'], 'content-type': 'application/json'}
        response = requests.post(url, headers=headers, data=payload)
        results = response.json().get('organic', [])
        top_results = results[:4]
        return '\n'.join([f"Title: {r['title']}\nLink: {r['link']}\nSnippet: {r['snippet']}\n-----------------" for r in top_results])

    @tool("Search competitors")
    def search_competitors(target_markets, industry):
        """Search for competitors based on target markets and industry."""
        query = f"companies in {industry} targeting {target_markets}"
        return SearchTools.search_internet(query)