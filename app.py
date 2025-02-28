import os
import uuid
import openai
import getpass
import random
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())

import streamlit as st

from IPython.display import display, Markdown,Image as IPythonImage
from PIL import Image


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o",api_key = openai.api_key,temperature=0)



def display_saved_plot(plot_path: str) -> None:
    """
    Loads and displays a saved plot from the given path in a Streamlit app.

    Args:
        plot_path (str): Path to the saved plot image.
    """
    if os.path.exists(plot_path):
        st.image(plot_path, caption="Saved Plot", use_column_width=True)
    else:
        st.error(f"Plot not found at {plot_path}")


##BI NODE...############################################################################################################
##BI NODE...############################################################################################################
## Setting up API key and environment related imports

import os
import re
import openai
import getpass
import random
import uuid
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage,ToolMessage, AIMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_core.tools import tool

from IPython.display import display, Markdown,Image as IPythonImage
from PIL import Image

# from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo",api_key = openai.api_key,temperature=0)
from config import llm,display_saved_plot


PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

#DEFINING HELPER FUNCTIONS.......#########################################################################################################################################
def get_prompt_file(data_source):
    """Return the appropriate prompt file based on the data source."""
    prompt_mapping = {
        'Outbound_Data.csv': 'Prompts/Prompt1.txt',
        'Inventory_Batch.csv': 'Prompts/Prompt2.txt',
        'Inbound_Data.csv': 'Prompts/Prompt3.txt'
    }
    return prompt_mapping.get(data_source)


def extract_code_segments(response_text):
    """Extract code segments from the API response using regex."""
    segments = {}
    
    # Extract approach section
    approach_match = re.search(r'<approach>(.*?)</approach>', response_text, re.DOTALL)
    if approach_match:
        segments['approach'] = approach_match.group(1).strip()
    
    # Extract content between <code> tags
    code_match = re.search(r'<code>(.*?)</code>', response_text, re.DOTALL)
    if code_match:
        segments['code'] = code_match.group(1).strip()
    
    # Extract content between <chart> tags
    chart_match = re.search(r'<chart>(.*?)</chart>', response_text, re.DOTALL)
    if chart_match:
        segments['chart'] = chart_match.group(1).strip()
    
    # Extract content between <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    if answer_match:
        segments['answer'] = answer_match.group(1).strip()
    
    return segments


@tool
def execute_analysis(df, response_text):
    """Execute the extracted code segments on the provided dataframe and store formatted answer."""
    results = {
        'approach': None,
        'answer': None,
        'figure': None,
        'code': None,
        'chart_code': None
    }
    
    try:
        # Extract code segments
        segments = extract_code_segments(response_text)
        
        if not segments:
            print("No code segments found in the response")
            return results
        
        # Store the approach and raw code
        if 'approach' in segments:
            results['approach'] = segments['approach']
        if 'code' in segments:
            results['code'] = segments['code']
        if 'chart' in segments:
            results['chart_code'] = segments['chart']
        
        # Create a single namespace for all executions
        namespace = {'df': df, 'pd': pd, 'plt': plt, 'sns': sns}
        
        # Execute analysis code and answer template
        if 'code' in segments and 'answer' in segments:
            # Properly dedent the code before execution
            code_lines = segments['code'].strip().split('\n')
            # Find minimum indentation
            min_indent = float('inf')
            for line in code_lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            # Remove consistent indentation
            dedented_code = '\n'.join(line[min_indent:] if line.strip() else '' 
                                    for line in code_lines)
            
            # Combine code with answer template
            combined_code = f"""
{dedented_code}

# Format the answer template
answer_text = f'''{segments['answer']}'''
"""
            exec(combined_code, namespace)
            results['answer'] = namespace.get('answer_text')
        

        if 'chart' in segments:
            # Properly dedent the chart code
            if "No" in segments['chart']:
                pass
            else:
                chart_lines = segments['chart'].strip().split('\n')
                chart_lines = [x for x in chart_lines if 'plt.show' not in x]
                # Find minimum indentation
                min_indent = float('inf')
                for line in chart_lines:
                    if line.strip():  # Skip empty lines
                        indent = len(line) - len(line.lstrip())
                        min_indent = min(min_indent, indent)
                # Remove consistent indentation
                dedented_chart = '\n'.join(line[min_indent:] if line.strip() else '' 
                                         for line in chart_lines)

                plot_path = os.path.join(PLOT_DIR, f"plot_{uuid.uuid4().hex}.png")

                # Append savefig logic to the dedented chart code to ensure the figure is saved
                dedented_chart += f"\nplt.savefig('{plot_path}', bbox_inches='tight')"

    
                exec(dedented_chart, namespace)
    
                results['figure'] = plot_path
    


        
        return results
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        return results
    


#DEFINING BI AGENT CLASS.......#########################################################################################################################################
class BIAgent_Class:
    def __init__(self,llm, prompt, tools, data_description, dataset, helper_functions=None):
        """
        Initialize an Agent with the required properties.

        Parameters:
        - prompt (str): The prompt that defines the agent's task or behavior.
        - tools (list): The tools that the agent has access to (e.g., APIs, functions, etc.)
        - data_description (str): A description of the dataset the agent will work with.
        - dataset (dict or pd.DataFrame): The dataset that the agent will use.
        - helper_functions (dict, optional): A dictionary of helper functions specific to the agent.
        """
        self.llm = llm
        self.prompt = prompt
        self.tools = tools
        self.data_description = data_description
        self.dataset = dataset
        self.helper_functions = helper_functions or {}

    def add_helper_function(self, name, func):
        """
        Add a helper function specific to this agent.

        Parameters:
        - name (str): The name of the function.
        - func (function): The function to be added.
        """
        self.helper_functions[name] = func

    def run(self, question):
        """
        Run the agent's task using the provided question, available tools, and helper functions.

        Parameters:
        - question (str): The question the agent needs to answer or solve.

        Returns:
        - str: The result of the agent's task.
        """
        
        # Define the prompt with placeholders for variables
        prompt_temp = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt.strip()),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        
        # Use the prompt to formulate the response
        result = llm.invoke(prompt_temp.invoke({"data_description":self.data_description,
                            "question":question,
                            "messages":[HumanMessage(content=question)]}))

        return result

    def generate_response(self, question):
        """
        Generate a response using the agent's prompt and data description.

        Parameters:
        - question (str): The question to be answered.

        Returns:
        - str: The generated response based on the prompt and dataset.
        """
        result = self.run(question)
        response = self.helper_functions['execute_analysis'].invoke({"df":self.dataset,"response_text":result.content})
#        message = response['approach']+"\nSolution we got from this approach is:\n"+response['answer']
#         answer = response['answer']
        return response




    def __repr__(self):
        """
        String representation of the agent, showing essential properties.
        """
        return f"Agent(prompt={self.prompt}, tools={self.tools}, data_description={self.data_description}, dataset={self.dataset.head()})"
    

##COST OPTIMIZATION NODE...############################################################################################################
##COST OPTIMIZATION NODE...############################################################################################################


## Setting up API key and environment related imports

import os
import re
import uuid 
import openai
import getpass
import random
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown,Image as IPythonImage
from PIL import Image

## LangChain related imports

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage,ToolMessage, AIMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_core.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent


from config import llm, display_saved_plot
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo",api_key = openai.api_key,temperature=0)


from cost_optimization import (
    llm, get_parameters_values, consolidate_shipments, calculate_metrics,
    analyze_consolidation_distribution, get_filtered_data
)



# Defining Helper Functions....#######################################################################################################################

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def execute_plot_code(plot_code: str,df) -> list:
    """
    Executes the provided plotting code. It searches for every occurrence of plt.show() in the code and
    replaces it with a plt.savefig() call that saves the current figure to a unique file.
    If no plt.show() is present and no plt.savefig() exists, it appends a plt.savefig() call at the end.
    
    Args:
        plot_code (str): The code to generate one or more plots.
    
    Returns:
        list: A list of file paths where the plots were saved.
    """
    plot_paths = []
    
    # Function to replace each plt.show() with a unique plt.savefig() call
    def replace_show(match):
        new_path = os.path.join(PLOT_DIR, f"plot_{uuid.uuid4().hex}.png")
        plot_paths.append(new_path)
        # Ensure proper indentation is preserved (using the match group 1)
        indent = match.group(1) if match.lastindex and match.group(1) else ""
        return f"{indent}plt.savefig('{new_path}', bbox_inches='tight')"
    
    # Replace all occurrences of plt.show() with unique plt.savefig() calls
    sanitized_code = re.sub(r'(^\s*)plt\.show\(\)', replace_show, plot_code, flags=re.MULTILINE)
    
    # If no plt.show was found and no plt.savefig exists in the code, append one at the end
    if not re.search(r"plt\.savefig", sanitized_code):
        new_path = os.path.join(PLOT_DIR, f"plot_{uuid.uuid4().hex}.png")
        sanitized_code += f"\nplt.savefig('{new_path}', bbox_inches='tight')"
        plot_paths.append(new_path)
    
#     print("Sanitized Code:\n", sanitized_code)
    
    exec_globals = {"df": df, "sns": sns, "plt": plt}
    try:
        exec(sanitized_code, exec_globals)
        plt.close('all')
    except Exception as e:
        return [f"Error generating plot: {e}"]
    
    return plot_paths



def extract_plot_code(intermediate_steps: list) -> tuple:
    """
    Extracts the plotting code from the agent's intermediate steps.
    
    Args:
        intermediate_steps (list): Intermediate steps from the agent response.
    
    Returns:
        tuple: (plot_code, response, thought)
    """
    for step in intermediate_steps:
        artifacts, _ = step
        
        # The agent's intermediate steps may contain the code under a key like 'tool_input'
        tool_input_ = artifacts.tool_input
        agent_message = artifacts.log
        
        # Extract plot code (everything after "```python" and before "```")
        match = re.search(r"```python(.*?)```", tool_input_, re.DOTALL)
        plot_code = match.group(1).strip() if match else None
        
        # Extract message (everything before "Thought:")
        response_match = re.search(r'^(.*?)\s*Thought:', agent_message, re.DOTALL)
        response = response_match.group(1).strip() if response_match else agent_message.strip()
        
        # Extract thought (text between "Thought:" and "Action:")
        thought_match = re.search(r'Thought:\s*(.*?)\s*Action:', agent_message, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
    return plot_code, response, thought

def agent_wrapper(agent_response: dict,df) -> dict:
    """
    Wraps the agent response to extract, execute, and display plotting code for each intermediate step.
    For each step, any generated plots are saved using unique file names.
    
    The final output is constructed to show:
      - Step 1 message
      - Step 1 plot paths
      - Step 2 message
      - Step 2 plot paths
      - ...
      - Final agent response
    
    Args:
        agent_response (dict): Response from the agent.
    
    Returns:
        dict: Contains the agent input, a list of step outputs (each with a message and plot paths),
              and a final_answer string combining all.
    """
    intermediate_steps = agent_response.get("intermediate_steps", [])
    step_outputs = []
    
    for step in intermediate_steps:
        artifacts, _ = step
        tool_input_ = artifacts.tool_input
        agent_log = artifacts.log
        
        # Extract the plotting code from the tool_input
        match = re.search(r"```python(.*?)```", tool_input_, re.DOTALL)
        plot_code = match.group(1).strip() if match else None
        plot_code = plot_code if "plt.show" in plot_code else None
        
        # Extract the message (everything before "Thought:") and optional thought
        message_match = re.search(r'^(.*?)\s*Thought:', agent_log, re.DOTALL)
        message = message_match.group(1).strip() if message_match else agent_log.strip()
        thought_match = re.search(r'Thought:\s*(.*?)\s*Action:', agent_log, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        full_message = message + ("\n" + thought if thought else "")
        
        # Execute the plotting code and get a list of plot paths
        plot_paths = execute_plot_code(plot_code,df) if plot_code else []
        
        step_outputs.append({
            "message": full_message,
            "plot_paths": plot_paths
        })
    
    # Build the final answer by interleaving messages and the list of plot paths
    final_message = ""
    for idx, step in enumerate(step_outputs, 1):
        final_message += f"Step {idx} Message:\n{step['message']}\n"
        if step['plot_paths']:
            for i, path in enumerate(step['plot_paths'], 1):
                final_message += f"Step {idx} Plot {i}: {path}\n"
        else:
            final_message += f"Step {idx} Plot: No plot generated.\n"
    
    final_agent_response = agent_response.get("output", "")
#     final_message += "\nFinal Agent Response:\n" + final_agent_response
    
    return {
        "input": agent_response.get("input"),
        "steps": step_outputs,
        "final_answer": final_agent_response
    }



# Creating Cost Optimization Agent Class....#######################################################################################################################

class AgenticCostOptimizer:
    def __init__(self, llm, parameters):
        """
        Initialize the Agentic Cost Optimizer.

        :param llm: The LLM model to use for queries.
        :param parameters: Dictionary containing required parameters.
        """
        self.llm = llm
        self.parameters = parameters
        self.df = parameters.get("df", pd.DataFrame())
        self.shipment_window_range = (1, 10)
        self.total_shipment_capacity = 36
    
    def get_filtered_df_from_question(self):
        """Extracts filtered data based on user query parameters."""
        group_field = 'SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME'
        df = self.parameters['df']
        df['SHIPPED_DATE'] = pd.to_datetime(df['SHIPPED_DATE'], dayfirst=True)
        
        df = get_filtered_data(self.parameters, df)
        if df.empty:
            raise ValueError("No data available for selected parameters. Try again!")
        return df
    
    def get_cost_saving_data(self):
        """Runs cost-saving algorithm and returns result DataFrame."""
        
        df = self.get_filtered_df_from_question()
        print("Shape of original data after filtering:",df.shape)
        
        df['GROUP'] = df['SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME']
        grouped = df.groupby(['PROD TYPE', 'GROUP'])
        date_range = pd.date_range(start=self.parameters['start_date'], end=self.parameters['end_date'])

        
        all_results = []
        for shipment_window in range(self.shipment_window_range[0], self.shipment_window_range[1] + 1):
            print(f"Consolidating orders for shipment window: {shipment_window}")
            all_consolidated_shipments = []
            for _, group_df in grouped:
                consolidated_shipments, _ = consolidate_shipments(
                    group_df, 0, 95, shipment_window, date_range, lambda: None, self.total_shipment_capacity
                )
                all_consolidated_shipments.extend(consolidated_shipments)
            
            metrics = calculate_metrics(all_consolidated_shipments, df)
            distribution, distribution_percentage = analyze_consolidation_distribution(all_consolidated_shipments, df)
            
            result = {
                'Shipment Window': shipment_window,
                'Total Orders': metrics['Total Orders'],
                'Total Shipments': metrics['Total Shipments'],
                'Total Shipment Cost': round(metrics['Total Shipment Cost'], 1),
                'Total Baseline Cost': round(metrics['Total Baseline Cost'], 1),
                'Cost Savings': metrics['Cost Savings'],
                'Percent Savings': round(metrics['Percent Savings'], 1),
                'Average Utilization': round(metrics['Average Utilization'], 1),
                'CO2 Emission': round(metrics['CO2 Emission'], 1)
            }
            all_results.append(result)
        
        self.parameters['all_results'] = pd.DataFrame(all_results)
    
    def consolidate_for_shipment_window(self):
        """Runs consolidation algorithm based on the selected shipment window."""
        df = self.get_filtered_df_from_question()
        df['GROUP'] = df['SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME']
        grouped = df.groupby(['PROD TYPE', 'GROUP'])
        date_range = pd.date_range(start=self.parameters['start_date'], end=self.parameters['end_date'])
        
        all_consolidated_shipments = []
        for _, group_df in grouped:
            consolidated_shipments, _ = consolidate_shipments(
                group_df, 0, 95, self.parameters['window'], date_range, lambda: None, self.total_shipment_capacity
            )
            all_consolidated_shipments.extend(consolidated_shipments)
        
        self.parameters['all_consolidated_shipments'] = pd.DataFrame(all_consolidated_shipments)
    
    def compare_before_and_after_consolidation(self):
        """Compares shipments before and after consolidation."""
        consolidated_df = self.parameters['all_consolidated_shipments']
        df = self.get_filtered_df_from_question()
        
        before = {
            "Days": df['SHIPPED_DATE'].nunique(),
            "Pallets Per Day": df['Total Pallets'].sum() / df['SHIPPED_DATE'].nunique(),
            "Pallets per Shipment": df['Total Pallets'].sum() / len(df)
        }
        after = {
            "Days": consolidated_df['Date'].nunique(),
            "Pallets Per Day": consolidated_df['Total Pallets'].sum() / consolidated_df['Date'].nunique(),
            "Pallets per Shipment": consolidated_df['Total Pallets'].sum() / len(consolidated_df)
        }
        
        percentage_change = {
            key: round(((after[key] - before[key]) / before[key]) * 100, 2) for key in before
        }
        
        comparison_df = pd.DataFrame({"Before": before, "After": after, "% Change": percentage_change})
        return comparison_df
    
    def handle_query(self, question):
            """Handles user queries dynamically, storing conversation history."""
            chat_history = []
            chat_history.append({"Human": question})

            result = get_parameters_values(self.parameters["api_key"], question)
            self.parameters.update(result)
            chat_history.append({"Agent": f"Parameters: {result}"})

            self.get_cost_saving_data()
            display(self.parameters['all_results'])
            max_savings_row = self.parameters['all_results'].loc[self.parameters['all_results']['Cost Savings'].idxmax()].to_dict()
            chat_history.append({"Agent": f"Optimum results: {max_savings_row}"})
            
            agent = create_pandas_dataframe_agent(self.llm, self.parameters['all_results'], verbose=False, 
                                                  allow_dangerous_code=True, handle_parsing_errors=True,
                                                 return_intermediate_steps=True)
            while True:
                user_input = input("Ask follow-up question or press Enter to proceed: ")
                if not user_input:
                    break
                else:
                    try:
                        response = agent.invoke(user_input)

#                         step = response.get("intermediate_steps")[0] # Since, there is only one step
#                         artifacts,output = step
#                         print("Log:\n",artifacts.log)
                        
                        response_ = agent_wrapper(response,self.parameters['all_results'])
        
                        for i,step in enumerate(response_['steps']):
                            print(f"Step {i+1}")
                            print(step['message'])
                            for plot in step['plot_paths']:
                                display_saved_plot(plot)
                        print(response_["final_answer"])
                                


                        if response_['plot_path']!='No plotting code found.':
                            display_saved_plot(response_['plot_path'])
                            
                        chat_history.append({"Human": user_input})
                        chat_history.append({"Agent": response_['final_answer']})
                        
#                         print("Agent Response:", response_['answer'])
                    except Exception as e:
                        print("Error processing follow-up question:",e)
                        print("Try reshaping your question.")
                        chat_history.append({"Human": user_input})
                        chat_history.append({"Agent": e})
                    
            user_window = input("Choose shipment window or press Enter to use default: ")
            self.parameters["window"] = int(user_window) if user_window else self.parameters['all_results']['Cost Savings'].idxmax()


            self.consolidate_for_shipment_window()
            display(self.parameters['all_consolidated_shipments'])

            agent = create_pandas_dataframe_agent(self.llm, self.parameters['all_consolidated_shipments'], verbose=False,
                                                  allow_dangerous_code=True,handle_parsing_errors=True,
                                                 return_intermediate_steps=True)
            while True:
                user_followup = input("Ask follow-up question or press Enter to proceed: ")
                if not user_followup:
                    break
                else:
                    try:
                        response = agent.invoke(user_followup)                        
                        response_ = agent_wrapper(response,self.parameters['all_consolidated_shipments'])
                        
                        for i,step in enumerate(response_['steps']):
                            print(f"Step {i+1}")
                            print(step['message'])
                            for plot in step['plot_paths']:
                                display_saved_plot(plot)
                        
                        print(response_["final_answer"])
                        
                        if response_['plot_path']!='No plotting code found.':
                            display_saved_plot(response_['plot_path'])
                            
                        chat_history.append({"Human": user_followup})
                        chat_history.append({"Agent": response_["final_answer"]})
                        
#                         print("Agent Response:", response_['answer'])
                        
                    except Exception as e:
                        print("Error processing follow-up question:", e)
                        print("Try rephrasing your question.")
                        chat_history.append({"Human": user_followup})
                        chat_history.append({"Agent": e})
            
            self.parameters['comparison'] = self.compare_before_and_after_consolidation()
            chat_history.append({"Agent": f"Before and After Consolidation comparison: {self.parameters['comparison'].to_dict()}"})

            return chat_history
    


##AGENTIC GRAPH...############################################################################################################
##AGENTIC GRAPH...############################################################################################################
## Setting up API key and environment related imports

import os
import re
import openai
import getpass
import random
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown


import warnings
warnings.filterwarnings("ignore")

## LangGraph related imports

from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import operator

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()


from uuid import uuid4
import functools
from typing import TypedDict, Annotated, List

## LangChain related imports

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage,ToolMessage, AIMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_core.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent


from config import llm,display_saved_plot
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo",api_key = openai.api_key,temperature=0)



################################################################################################################################################################
from BIAgent_Node import BIAgent_Class,execute_analysis,get_prompt_file
from supervisor import supervisor_chain,members
from CostOptimization_Node import AgenticCostOptimizer

################################################################################################################################################################




class AgentState(TypedDict):
#     messages: Annotated[list[AnyMessage], operator.add]
    messages : Annotated[list[AnyMessage], add_messages]
    next : str



def supervisor_node(state: AgentState,chain=supervisor_chain):
    result = supervisor_chain.invoke(state['messages'])
    
#     if result['next']!='FINISH':
#         return {"messages": AIMessage(content=f"Calling {result['next']}, for question: {state['messages']}. Please provide answer to this question."),"next":result['next']}
    
    return {"messages": [AIMessage(content=f"Calling {result['next']}.")],"next":result['next']}


def agent_node(state, agent, name):
    result = agent(state) # This result will be the list of messages
    return {"messages": result,"next":"supervisor"}


def bi_agent(state: AgentState):
    """BI Agent (Business Intelligence Agent) is responsible for analyzing shipment data to generate insights. 
    It handles tasks such as performing exploratory data analysis (EDA), calculating summary statistics, identifying trends, 
    comparing metrics across different dimensions (e.g., users, regions), and generating visualizations to help 
    understand shipment-related patterns and performance.
    """


    file_path = '/Users/ay/Desktop/GenAI-POC/Perrigo/Perrigo-GenAI-Answer-Bot'

    # Loading dataset
    data_source = 'Data/Outbound_Data.csv'
    data_source = os.path.join(file_path,data_source)
    df = pd.read_csv(data_source)

    # Loading data description prompt
    prompt_file_path= os.path.join(file_path,get_prompt_file(data_source.split('/')[-1]))
    with open(prompt_file_path, 'r') as file:
        data_description = file.read().strip()

    # Writing prompt to define role and give context about the task
    prompt = """
                            
    You are an AI assistant tasked with analyzing a dataset to provide code for calculating the final answer and generating relevant visualization.
    I will provide you with the data in dataframe format, as well as a question to answer based on the data.

    {data_description}

    Here is the question I would like you to answer using this data:
    <question>
    {question}
    </question>

    To answer this, first think through your approach inside <approach> tags. Break down the steps you
    will need to take and consider which columns of the data will be most relevant. Here is an example:
    <approach>
    To answer this question, I will need to:
    1. Calculate the total number of orders and pallets across all rows
    2. Determine the average distance and cost per order
    3. Identify the most common PROD_TYPE and SHORT_POSTCODE
    </approach>

    Then, write the Python code needed to analyze the data and calculate the final answer inside <code> tags. Assume input dataframe as 'df'
    Be sure to include any necessary data manipulation, aggregations, filtering, etc. Return only the Python code without any explanation or markdown formatting.
    For decimal answers round them to 1 decimal place.

    Generate Python code using matplotlib and/or seaborn to create an appropriate chart to visualize the relevant data and support your answer.
    For example if user is asking for postcode with highest cost then a relevant chart can be a bar chart showing top 10 postcodes with highest total cost arranged in decreasing order.
    Specify the chart code inside <chart> tags.
    When working with dates:

    Always convert dates to datetime using pd.to_datetime() with explicit format
    For grouping by month, use dt.strftime('%Y-%m') instead of dt.to_period()
    Sort date-based results chronologically before plotting

    The visualization code should follow these guidelines:

    Start with these required imports:

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    Use standard chart setup:
    # Set figure size and style
    plt.figure(figsize=(8, 5))
    # Set seaborn default style and color palette
    sns.set_theme(style="whitegrid")  
    sns.set_palette('pastel')

    For time-based charts:


    Use string dates on x-axis (converted using strftime)
    Rotate labels: plt.xticks(rotation=45, ha='right')
    Add gridlines: plt.grid(True, alpha=0.3)

    For large numbers:
    Format y-axis with K/M suffixes using:

    Always include:

    Clear title (plt.title())
    Axis labels (plt.xlabel(), plt.ylabel())
    plt.tight_layout() at the end


    For specific chart types:

    Time series: sns.lineplot() with marker='o'
    Rankings: sns.barplot() with descending sort
    Comparisons: sns.barplot() or sns.boxplot()
    Distributions: sns.histplot() or sns.kdeplot()

    Return only the Python code without any explanation or markdown formatting.

    Finally, provide the answer to the question in natural language inside <answer> tags. Be sure to
    include any key variables that you calculated in the code inside {{}}."""

    # Providing Helper Function
    helper_functions = {"execute_analysis":execute_analysis}

    
    # Finally Creating BI Agent Variable...
    BIAgent = BIAgent_Class(llm=llm, prompt=prompt, tools=[], data_description=data_description, dataset=df, 
                helper_functions=helper_functions)
    
   
    question = state['messages'][len(state['messages'])-2].content

    response = BIAgent.generate_response(question)
    
    if response['figure']:
        display_saved_plot(response['figure'])
        
    message = response['approach']+"\nSolution we got from this approach is:\n"+response['answer']
    
    

    return [HumanMessage(content=message)]


def cost_saving_agent(state: AgentState):
    """Cost Optimization Agent is responsible for analyzing shipment cost-related data and recommending 
    strategies to reduce or optimize costs. This agent handles tasks such as identifying cost-saving 
    opportunities, calculating the optimal number of trips, performing scenario-based cost optimizations 
    (e.g., varying consolidation windows, truck capacity adjustments), and providing benchmarks and comparisons
    between current and optimized operations. The agent also calculates key performance metrics like cost per 
    pallet, truck utilization rate, and cost savings over time. This agent is called when the user asks about 
    shipment cost reduction or optimization scenarios."""
    
    file_name = "Complete Input.xlsx"
    cost_saving_input_df = pd.read_excel('Complete Input.xlsx', sheet_name='Sheet1')
    query = state['messages'][len(state['messages'])-2].content
    
    
    parameters = {"api_key": openai.api_key, 
              "query": query, 
              "file_name":file_name,
              "df": cost_saving_input_df
             }
    
    agent = AgenticCostOptimizer(llm, parameters)
    conv = agent.handle_query(query)
    
    chat=[]
    for msg in conv:
        key, value = list(msg.items())[0]
        if "Agent" in key:
            if type(value) is not str:
                value = str(value)
            chat.append(AIMessage(content=value))
        else:
            chat.append(HumanMessage(content=value))
    
    result = llm.invoke(f"Summarise this conversation and provide a formatted response to the first question asked by human. Hers is the conversation: {chat}")
#     AIMessage(content=result.content)
    
    message = result.content
    return [HumanMessage(content=message)]


def generate_scenario_agent(state: AgentState):
    """Generate Scenario Agent is responsible for creating and analyzing "what-if" scenarios based on 
    user-defined parameters. This agent helps compare the outcomes of various decisions or actions, such as 
    the impact of increasing truck capacity, changing shipment consolidation strategies, or exploring different 
    operational scenarios. It can model changes in the system and assess the consequences of those changes to 
    support decision-making and optimization. This agent is called when the user asks about scenario generation,
    comparisons of different outcomes, or analysis of hypothetical situations."""
    
    message = 'Generate Scenario Agent Called..! This is your answer: <x></x>. Choose next as "FINISH"'
    return [HumanMessage(content=message)]

def driver_identification_agent(state: AgentState):
    """This function should be called if the user question is related
    to identify the drivers of cost of shipmemts."""
    

    message = f'Driver Identification Agent Called..! This is your answer: <x></x>. Choose next as "FINISH"'
    return [HumanMessage(content=message)]


def conversation_agent(state: AgentState) -> List[HumanMessage]:
    """Conversational agent for chat-based support and supervisor fallback."""
    
    conv=[]
    question = state['messages'][len(state['messages'])-2].content  # Extract user question
    conv.append("Human:")
    conv.append(question)
    while True:
        if not question:
            break
        else:

            response = llm.invoke(f"You are a helpful chat assitant. Use this conversation context: {" ".join(conv)} to answer the question.")  # Get LLM-generated response
            print(f"AI:\n{response.content}")
            conv.append("Conv Agent:")
            conv.append(response.content)
            question = input("User: \n")
            conv.append("Human:")
            conv.append(question)

    return [HumanMessage(content=" ".join(conv))]


################################################################################################################################################################

# Defining Agent Nodes..!
bi_agent_node = functools.partial(agent_node, agent=bi_agent, name="BI Agent")
driver_identification_agent_node = functools.partial(agent_node, agent=driver_identification_agent, name="Driver Identification Agent")
cost_saving_agent_node = functools.partial(agent_node, agent=cost_saving_agent, name="Cost Saving Agent")
generate_scenario_agent_node = functools.partial(agent_node, agent=generate_scenario_agent, name="Generate Scenario Agent")
conversation_agent_node = functools.partial(agent_node, agent=conversation_agent, name="Conversation Agent")

workflow = StateGraph(AgentState)
workflow.add_node("BI Agent", bi_agent_node)
workflow.add_node("Driver Identification Agent", driver_identification_agent_node)
workflow.add_node("Cost Saving Agent", cost_saving_agent_node)
workflow.add_node("Generate Scenario Agent", generate_scenario_agent_node)
workflow.add_node("Conversation Agent", conversation_agent_node)
workflow.add_node("supervisor", supervisor_node)

for member in members:
# We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member['agent_name'], "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k['agent_name']: k['agent_name'] for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.set_entry_point("supervisor")

multi_agent_graph = workflow.compile(checkpointer=memory)



#....APP.....#############################################################################################
#....APP.....#############################################################################################
#....APP.....#############################################################################################


question = "Hi ! I'm Stephen."
config = {"configurable":{"thread_id":"1"}}
state = {"messages":[HumanMessage(content=question)],"next":"supervisor"}
graph = multi_agent_graph

state['messages'][0].pretty_print()
counter=0
while state['next']!='FINISH' and counter<100:
    
    #Getting the current state
    current_state= graph.nodes[state['next']].invoke(state,config)
    current_state['messages'][0].pretty_print()

    # Updating the state
    state['messages']=add_messages(state['messages'],current_state['messages'])
    state['next'] = current_state['next']
    counter+=1


    
#....END.....#############################################################################################