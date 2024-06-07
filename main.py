from crewai import Crew
from textwrap import dedent
from dotenv import load_dotenv

from research_agent import TaskAgents
from research_tasks import GenericProblemSolvingTasks

# Load environment variables from .env file
load_dotenv()

class TaskWeaver:
    def __init__(self, query):
        self.query = query

    def run(self):
        agents = TaskAgents()
        tasks = GenericProblemSolvingTasks()

        planner = agents.planner_agent()
        executor = agents.executor_agent()

        planner_task = tasks.research(planner, self.query)

        crew = Crew(
            agents=[planner],
            tasks=[planner_task],
            verbose=True
        )

        # Run the planning task to get the list of tasks
        result = crew.kickoff()
        planned_tasks = result  # Assume the result contains a 'tasks' key with the planned tasks
        print(planned_tasks)

        # # Create execution tasks based on the planned tasks
        # executor_tasks = tasks.execute_task(executor, planned_tasks)

        # crew = Crew(
        #     agents=[planner, executor],
        #     tasks=[planner_task],
        #     verbose=True
        # )

        # # Update the crew with the new tasks
        # crew.tasks.extend(executor_tasks)

        # # Run the execution tasks
        # final_result = crew.kickoff()
        return planned_tasks


if __name__ == "__main__":
    print("## Welcome to Taskweaver")
    query = input(dedent("How can i help you?\n"))

    task_weaver = TaskWeaver(query)
    result = task_weaver.run()
    print("\n\n########################")
    print("## Here is the Report")
    print("########################\n")
    print(result)