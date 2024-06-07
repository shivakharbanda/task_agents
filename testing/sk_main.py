from langchain_openai import AzureChatOpenAI
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

# Define the Pydantic model for the expected output
class Task(BaseModel):
    task: str = Field(description="Description of the task.")
    order: int = Field(description="Order of the task.")
    dependencies: Optional[List[int]] = Field(description="List of task order numbers that this task depends on.", default=[])

class TaskBreakdown(BaseModel):
    tasks: List[Task]

# Initialize the model with your Azure settings
llm = AzureChatOpenAI(
    api_key=os.environ['AZURE_API_KEY'],
    api_version=os.environ['AZURE_API_VERSION'],
    azure_endpoint=os.environ['AZURE_ENDPOINT'],
    azure_deployment=os.environ['AZURE_DEPLOYMENT_NAME']
)

# Define few-shot examples
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

# Define the human message
human_message = HumanMessage(content="Show me vision and mission for Scikiq data and perform competition benchmarking for data fabric platform against Scikiq. Once benchmarking is done, produce a SWOT report. Also answer, why customer should buy SCIKIQ than competition?")

# Define structured output model
structured_llm = llm.with_structured_output(TaskBreakdown)

messages = [
    SystemMessage(content=system_message),
    HumanMessage(content=human_message.content),
]

# Invoke the model using the combined prompt and print the result
result = llm.invoke(messages)

# Wrap the response in a valid JSON format
messages = [
    SystemMessage(content="Your job is to put the response into a valid json format and return it."),
    HumanMessage(content=result.content),
]

result = llm.invoke(messages)

print(result.content)
