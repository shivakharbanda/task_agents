# input main_query_nl

# query_breakdown_agent(main_query_nl) -> (subqueries_nl)

# query_cleaner_agent(subqueries_nl) -> (subqueries_cleaned)

# for subquery in subqueries_cleaned:

#     fetch_results_and_clean(subquery) -> (temp_cleaned)

#     subquery_results[subquery] = temp_cleaned

# research_interpreter_agent(main_query_nl, subquery_results)


from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks import get_openai_callback
import requests
import json
from bs4 import BeautifulSoup
from task_agents.tools.browser_tools import BrowserTools

load_dotenv()

examples = [
    {
        "task": "Plan a birthday party",
        "output": {
            "tasks": [
                {"task": "Choose a theme", "order": 1, "dependencies": []},
                {"task": "Make a guest list", "order": 2, "dependencies": [1]},
                {"task": "Send invitations", "order": 3, "dependencies": [2]},
                {"task": "Buy decorations", "order": 4, "dependencies": [1]},
                {"task": "Set up venue", "order": 5, "dependencies": [4]}
            ]
        }
    },
    {
        "task": "Organize a conference",
        "output": {
            "tasks": [
                {"task": "Choose a date", "order": 1, "dependencies": []},
                {"task": "Book a venue", "order": 2, "dependencies": [1]},
                {"task": "Send invitations", "order": 3, "dependencies": [2]},
                {"task": "Arrange speakers", "order": 4, "dependencies": [3]},
                {"task": "Prepare materials", "order": 5, "dependencies": [4]}
            ]
        }
    }
]

# Create the example strings for the system message
example_strings = "\n".join([
    f"example_user: {example['task']}\nexample_assistant: {example['output']}"
    for example in examples
])

# Define the system message with few-shot examples
system_message = f"""
You are an assistant that breaks down complex tasks into smaller, manageable sub-tasks.
Your task is to break down the provided problem into multiple subproblems which can be solved in sequence to address the bigger problem.
JUST RETURN THE OUTPUT JSON WITH THE TASK BREAKDOWN AND NOTHING ELSE.
The JSON output should be in the following format:
{{
  "tasks": [
    {{
      "task": "Description of the task",
      "order": Order number,
      "dependencies": [List of task order numbers that this task depends on]
    }},
    ...
  ]
}}

Here are some examples of task breakdowns:

{example_strings}

When the user enters their input, provide no explanation and only return a valid JSON response.

See below the instruction for creating the tasklist. 
1. Identify Task/action verbs from the user query.
2. For each identified task, break the user query into multiple sub-queries/tasks in the format of Subject, Object, and Verb triplet.
3. Arrange these tasks in logical order and add it back to tasklist.
4. Create independent tasks as parallel and dependent as sequential and in the right order.
"""


class NLPAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            api_key=os.environ['AZURE_API_KEY'],
            api_version=os.environ['AZURE_API_VERSION'],
            azure_endpoint=os.environ['AZURE_ENDPOINT'],
            azure_deployment=os.environ['AZURE_DEPLOYMENT_NAME'],
        )
    def query_simplifier(self, message):
        response = self.llm.invoke([
            SystemMessage(content="The user will give you a piece of information he needs. You need to turn his natural language into a query that google can effectively answer. Remember, Google needs to answer the query, not you, so if you feel you don't recognize something, just make a query to the best of your abilities, but do not start asking the user questions. Do not start with an introduction or acknowledge the task. When the user enters their query, you must simply respond with the google query on the first line and nothing else. Don't include bullet points or dashes either."),
            AIMessage(content="Understood. Please enter your query."),
            HumanMessage(content=message)
        ])
        return response.content
    def query_breakdown(self, message):
        response = self.llm.invoke([
                SystemMessage(content="You are a useful assistant. The user will give you a certain query, and you must identify what information you need. From that, generate multiple elementary queries which a search engine like google can return results for. When the user enters their input, provide no explanation and write no introduction. You must only return a each subquery on a new line and nothing else, don't even add bullet points or dashes."),
                HumanMessage(content=message)
            ])
        return response.content.split("\n")
    def subquery_sequencer(self, subq, mainquery):
        response = self.llm.invoke([
                SystemMessage(content="You are a useful assistant. Person A said the following: " + mainquery + "\nYou must help the user reply to person A. The user will tell you some information they need to be able to answer person A effectively. You need to identify whether you need to create multiple google searches to find the information you need to address what the user asked for. For example, if the query requires comparison of something to something else, maybe the first query could be to search for a broader perspective and then the second query could be for the main subject. From the information the user needs, generate multiple elementary queries which a search engine like google can return results for. When the user enters their input, provide no explanation and write no introduction. You must only return a each subquery on a new line and nothing else, don't even add bullet points or dashes."),
                HumanMessage(content=subq)
            ])
        return response.content.split("\n")
    def custom_pass(self, custommsgs):
        response = self.llm.invoke(custommsgs)
        return response.content
    def summarize_website(self, website, mainquery):
        try:
            response = requests.get(website)
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
            totaltext = ''
            for curtext in content:
                totaltext += curtext.text.strip()
            lenofpage = len(totaltext)
            timestoit = lenofpage // 5000 + 1
            allsummaries = []
            jumps = 10000
            keycontent = []
            for i in range(timestoit):
                if i * jumps + jumps < len(totaltext):
                    curextract = totaltext[i * jumps: i * jumps + jumps]
                else:
                    curextract = totaltext[i * jumps: ]
                response = self.llm.invoke([
                        SystemMessage(content='Here are some statements made by Person A: ' + mainquery + '\n\nThe user will give you some text, you have to analyze and summarize it and find key quotes that relate to or help answer Person As statement. Dont start with an introduction or acknowledge the task, simply give the key components and analysis/summary in the form of a json dict like this {"key_quotes":[], "summary":""}. Make sure to keep key information for whatever is relevant to the statements made by Person A.'),
                        AIMessage(content="Understood. Can you provide the text now that I have to summarize?"),
                        HumanMessage(content=curextract)
                    ])
                curresponse = response.content
                responsedict = json.loads(curresponse)
                keycontent += responsedict['key_quotes']
                allsummaries.append(responsedict['summary'])

            mainreturn = self.summarize_paragraphs(allsummaries, mainquery)
            return keycontent, mainreturn
        except:
            return [], ''
            
    def summarize_paragraphs(self, paragraphs, mainquery, keycontent=[]):
        fulltext = ''
        for paragraph in paragraphs:
            fulltext += paragraph
        mainlist = [
            SystemMessage("The user will give you a few paragraphs. You must identify any redundancies or useless information and remove it. You must also make the text clearer and more cohesive and summarize it somewhat but keep the main content.Make sure to keep whatever is relevant to the following statements made by Person A: \n\n" + mainquery),
            HumanMessage(fulltext)
        ]
        if keycontent != []:
            mainlist.append(HumanMessage("Also, here's some key quotes you might or might not need: " + str(keycontent)))
        response = self.llm.invoke(mainlist)
        return response.content
    
    def generate_final_answer(self, mainquery, subq_content_map):
        infostr = ''
        for subq in subq_content_map:
            infostr += subq + ': ' + subq_content_map[subq] + '\n'
        starterstr = 'The user is going to give you some information to generate a response to the following thing Person A said: \n\n' + mainquery + '\n\n The user will have broken down the information you might need into different pieces. Use the information you think is relevant from what the user provides to give your final answer to the thing Person A said.'
        response = self.llm.invoke([
            SystemMessage(starterstr),
            HumanMessage(infostr)
        ])
        return response.content

class WebAgent:
    def __init__(self):
        self.video_domains = ["youtube.com", "vimeo.com", "dailymotion.com"]
    def search_term(self, subquery, num_results=20):
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": subquery})
        headers = {'X-API-KEY': os.environ['SERPER_API_KEY'], 'content-type': 'application/json'}
        response = requests.post(url, headers=headers, data=payload)
        results = response.json().get('organic', [])
        results = [url['link'] for url in results if not any(domain in url['link'] for domain in self.video_domains)]
        if len(results) > 4:
            results = results[:3]
        return results

QueryBreakdownAgent = NLPAgent()

checking = BrowserTools()

def main():
    NLPAgentMain = NLPAgent()
    WebAgentMain = WebAgent()
    main_query = input("Enter main query: ")
    subqueries = NLPAgentMain.query_breakdown(main_query)
    print(subqueries)
    subq_url_map = {}
    subq_answer_map = {}
    for subquery in subqueries:
        subsubqueries = NLPAgentMain.subquery_sequencer(subquery, main_query)
        for subsubquery in subsubqueries:
            subq_url_map[subsubquery] = WebAgentMain.search_term(subquery)
            print('searching', subsubquery)
            print(subq_url_map[subsubquery])

    for subq in subq_url_map:
        allsitesummaries = []
        for website in subq_url_map[subq]:
            curr_site_summary = checking.partition_and_summarize(checking.scrape_website_content(website))
            allsitesummaries.append(curr_site_summary)
        print('summarizing paras')
        subq_answer_map[subq] = NLPAgentMain.summarize_paragraphs(allsitesummaries, main_query, keycontent)

    print('generating final answer')
    final = NLPAgentMain.generate_final_answer(main_query, subq_answer_map)
    print(final)
        
