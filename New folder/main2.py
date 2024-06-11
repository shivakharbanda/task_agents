from main import NLPAgent, WebAgent
from task_agents.tools.browser_tools import BrowserTools
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


taskresults = {}

tasklist = {'tasks': [
    {'task': 'Find the vision for Scikiq data', 'order': 1, 'dependencies': [], 'needs_google': True},
    {'task': 'Find the mission for Scikiq data', 'order': 2, 'dependencies': [], 'needs_google': True},
    {'task': 'Identify competitors in the data fabric platform space', 'order': 3, 'dependencies': [], 'needs_google': True},
    {'task': 'Benchmark Scikiq against competitors', 'order': 4, 'dependencies': [1, 2, 3], 'needs_google': False},
    {'task': 'Produce a SWOT report for Scikiq', 'order': 5, 'dependencies': [2, 4], 'needs_google': False},
    {'task': 'Identify unique selling points of Scikiq over the competition', 'order': 6, 'dependencies': [5], 'needs_google': False}
]}

sorted_tasklist = sorted(tasklist['tasks'], key=lambda x: x['order'])

mainquery = "Show me vision and mission for scikiq data and perform competition benchmarking for data fabric platform against scikiq. Once benchmarking is done, produce a SWOT report. Also answer, why customer should buy SCIKIQ than competition?"

NLPMain = NLPAgent()
WebMain = WebAgent()
BrowserMain = BrowserTools()

temp = BrowserTools()
print(temp.scrape_and_summarize_website("scikiq.com", "Find the vision for Scikiq data"))

key_quotes = []

for id, task in enumerate(sorted_tasklist):
    if task['needs_google']:
        websites = WebMain.search_term(NLPMain.query_simplifier(task['task']).replace("\"", ""))
        print(websites)
        allsummaries = []
        for website in websites:
            key_quotes_temp, summary = BrowserMain.scrape_and_summarize_website(website, task['task'])
            key_quotes += key_quotes_temp
            allsummaries.append(summary)
        finaltaskresult = NLPMain.summarize_paragraphs(allsummaries, task['task'])
        taskresults[id + 1] = finaltaskresult
    if not task['needs_google']:
        totalpass = [SystemMessage(content="Can you answer the following query using information that the user gives you? The query is: " + task['task']),
                     AIMessage(content="Yes, I can. Go ahead and enter the information you would like me to use.")]
        context = ' '
        for dependency in task['dependencies']:
            # add the task and its output
            context += "\n" + task['task'] + " : " + taskresults[dependency] + "\n"
        totalpass.append(HumanMessage(content=context))
        cur_response = NLPMain.custom_pass(totalpass)
        taskresults[id + 1] = cur_response


mainpass = [SystemMessage("I want you to answer the following query:" + mainquery + "\n\n You must answer it using the information the user will give you next."),
                AIMessage("Understood. Please enter the information you would like me to respond using.")]
context = ''

for taskid in taskresults:
    context += taskresults[taskid] + "\n"

mainpass.append(HumanMessage(content=context))

finalresponse = NLPMain.custom_pass(mainpass)
print(finalresponse)
    
    

# you have to use the information that i provided here to solve this main task whose output depends on all the tasks i have mentioned
# add the task and ask for suitable response 
        
            
        