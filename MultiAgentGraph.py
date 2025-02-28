## Setting up API key and environment related imports

import os
import re
import uuid
import openai
import getpass
import random
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())

import streamlit as st

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
        
    # message = response['approach']+"\n\nSolution we got from this approach is:\n\n"+response['answer']
    
    message = response['answer']

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
    
    result = llm.invoke(f"This is the response provided by the Cost Optimization Agent: {chat}. Generate a final response to be shown to the client.")
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
    """Conversational agent for chat-based support and supervisor fallback.
    
    Processes a single conversation turn: it extracts the latest user question from the state,
    generates an LLM response based on the current conversation context, then reads a new user
    question from input. The entire conversation (current question, LLM response, and the new question)
    is returned as a single HumanMessage.
    """
    conv = []
    
    # Extract the latest user question from the state (assumes the second-to-last message is the user question)
    question = state['messages'][len(state['messages']) - 2].content
    conv.append("Human:")
    conv.append(question)
    
    # Generate the LLM response using the current conversation context
    conversation_context = " ".join(conv)
    response = llm.invoke(
        f"You are a helpful chat assistant. Use this conversation context: {conversation_context} to answer the question."
    )
    conv.append("Conv Agent:")
    conv.append(response.content)
    st.write("\n".join(conv[-2:]))
    
    # Get a new user question
    # new_question = input("User: \n")
    new_question = st.text_input("User: \n",key=uuid.uuid4().hex)
    if st.button("Submit",key=uuid.uuid4().hex):
        if not question:
            st.error("Please enter a question before submitting")
            return
    
        conv.append("Human:")
        conv.append(new_question)
    
    # Return the entire conversation as a single HumanMessage
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



