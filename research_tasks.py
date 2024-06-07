from crewai import Task
from textwrap import dedent

class GenericProblemSolvingTasks:
    
    def research(self, agent, problem_description):
        return Task(
            description=dedent(f"""
                Analyze the provided problem description and generate a comprehensive 
                plan. This includes identifying all necessary tasks and determining 
                the order in which they need to be performed. The plan should be detailed 
                and clear for execution.

                Problem Description: {problem_description}

                Your final plan MUST include:
                - A list of all tasks required to solve the problem.
                - The order in which these tasks should be performed.
                - Any dependencies between tasks.

                Ensure to cover all aspects of the problem and provide a clear roadmap 
                for execution.
                return the ouput in json only the json nothing else
            """),
            agent=agent,
            expected_output="A detailed plan with a list of tasks, their order, and any dependencies."
        )

    def execute_task(self, agent, task_description):
        return Task(
            description=dedent(f"""
                Execute the given task using the best available tools. If a suitable tool 
                is not found, identify and describe the required tool and raise an exception 
                to end the process. The executor cannot delegate the tasks further.

                Task: {task_description}

                Your final output MUST include:
                - The results of the executed task.
                - Any issues encountered and how they were resolved.
                - If a suitable tool was not found, describe the needed tool and raise an exception.

                Ensure to use the most effective and efficient tools available to complete the task.
            """),
            agent=agent,
            expected_output="Results of the executed task, resolution of issues, or a description of the needed tool if not found."
        )