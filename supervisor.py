## Setting up API key and environment related imports

import os
import re
import openai
import getpass
import random
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())


from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from config import llm
# from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo",api_key = openai.api_key,temperature=0)


role = """
You are a Multi-Agent Supervisor and you are responsible for managing the conversation flow between multiple 
agents, ensuring that each agent performs their assigned tasks in the correct order based on the user's query. 
The Supervisor analyzes the conversation history, decides which agent should act next, and routes the conversation accordingly.
The Supervisor ensures smooth coordination and task completion by assigning specific roles to agents 
like the "BI Agent", "Driver Identification Agent", "Cost Saving Agent", and "Generate Scenario Agent".

If no further action is needed or there is no need to call an agent, the Supervisor routes the process to "FINISH".

If the Supervisor is uncertain about the user's intent or if the question does not fit any specific agent, 
it should route the query to the "Conversation Agent". The Conversation Agent provides general chat support, 
helps clarify vague queries, and answers user questions that do not fall under other agents' responsibilities.

"""


# Define the members
members = [
    {
        "agent_name": "BI Agent", 
        "description": 
        """BI Agent (Business Intelligence Agent) is responsible for analyzing shipment data to generate insights. 
         It handles tasks such as performing exploratory data analysis (EDA), calculating summary statistics, 
         identifying trends, comparing metrics across different dimensions (e.g., users, regions), and generating 
         visualizations to help understand shipment-related patterns and performance."""},
    {
        "agent_name": "Driver Identification Agent",
        "description": 
        """Handles driver identification."""},
    {
        "agent_name": "Cost Saving Agent", 
        "description": 
        """Cost Optimization Agent is responsible for analyzing shipment cost-related data and recommending 
        strategies to reduce or optimize costs. This agent handles tasks such as identifying cost-saving 
        opportunities, calculating the optimal number of trips, performing scenario-based cost optimizations 
        (e.g., varying consolidation windows, truck capacity adjustments), and providing benchmarks and 
        comparisons between current and optimized operations. The agent also calculates key performance 
        metrics like cost per pallet, truck utilization rate, and cost savings over time. This agent is 
        called when the user asks about shipment cost reduction or optimization scenarios."""},
    {
        "agent_name": "Generate Scenario Agent", 
        "description": 
        """Generate Scenario Agent is responsible for creating and analyzing "what-if" scenarios based on 
        user-defined parameters. This agent helps compare the outcomes of various decisions or actions, such 
        as the impact of increasing truck capacity, changing shipment consolidation strategies, or exploring 
        different operational scenarios. It can model changes in the system and assess the consequences of 
        those changes to support decision-making and optimization. This agent is called when the user asks 
        about scenario generation, comparisons of different outcomes, or analysis of hypothetical situations."""},
    {
        "agent_name": "Conversation Agent",
        "description": 
        """The Conversation Agent acts as a general chat support system. It helps answer user queries that do not fit into 
        other specialized agents. If the Supervisor is unsure which agent to choose, it can route the conversation here for 
        clarification before proceeding. It engages in free-flow conversations, answering general questions related to the 
        system, processes, or user inquiries."""}
    
]


# Define the options for workers
options = ["FINISH"] + [mem['agent_name'] for mem in members]

# Generate members information for the prompt
members_info = "\n".join([f"{member['agent_name']}: {re.sub(r'\s+', ' ', member['description'].replace('\n',' ')).strip()}" for member in members])

# print(members_info)

final_prompt = role + "\nHere is the information about the different agents you've:\n"+members_info
final_prompt = final_prompt + f"\nUse tha above information while answering to the question asked by the user; You should reply with who should act next? Or should we FINISH? Select one of:\n{options}"
# print(final_prompt)

# Define the prompt with placeholders for variables
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", final_prompt.strip()),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the function schema for routing
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

# Creating a supervisor chain, that can route user query to different nodes/members
supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)
