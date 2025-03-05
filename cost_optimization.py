import os
import re
import openai
import getpass
import random
import ast

from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from itertools import product
from collections import Counter, defaultdict


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image
from pyecharts import options as opts
from pyecharts.charts import Calendar, Page
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode
import streamlit.components.v1 as components
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, ColorBar
from bokeh.palettes import RdYlGn11
from bokeh.transform import linear_cmap



## LangChain related imports
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage,ToolMessage, AIMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage


## Model/LLM related imports
# from langchain_openai import ChatOpenAI
from config import llm


############################################################################################################################################################################################
def get_chatgpt_response(api_key, instructions, user_query):
    """
    Sends a query to OpenAI's ChatCompletion API with the given instructions and user query.
    """
    # Set the API key
    # openai.api_key = api_key
 
    try:
        # Send the query to OpenAI ChatCompletion API
        response = openai.chat.completions.create(
            model="gpt-4o",  # Specify the GPT-4 model
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_query}
            ],
            max_tokens=500,  # Adjust token limit based on your needs
            temperature=0.7  # Adjust for creativity (0.7 is a balanced value)
        )
        # Extract and return the assistant's response
        return response.choices[0].message.content

    except openai.OpenAIError as e:
        # Handle OpenAI-specific errors
        return f"An error occurred with the OpenAI API: {str(e)}"

    except Exception as e:
        # Handle other exceptions (e.g., network issues)
        return f"An unexpected error occurred: {str(e)}"


def ask_openai(selected_customers,selected_postcodes,customers,postcodes):
    """
    Sends a question and data context to OpenAI API for processing.
    """
    # Formulate the prompt
    prompt = f"""You are given four lists as inputs:

    {selected_customers}: A list of selected customer names.
    {selected_postcodes}: A list of selected postcodes.
    {customers}: A list of unique customer names.
    {postcodes}: A list of unique postcodes corresponding to the customers.
    
    Your task is to find the best match for each item in {selected_customers} and {selected_postcodes} from the {customers} and {postcodes} lists respectively. The matching should be case-insensitive and prioritize similarity. If there are multiple possible matches, return the most suitable one.

    The output should consist of two separate lists:

    A list of matched customers.
    A list of matched postcodes.

    Example Input:
    selected_customers = ['Alloga', 'FORum', 'usa']  
    selected_postcodes = ['ng', 'Lu']  
    customers = ['ALLOGA UK', 'FORUM', 'USA', 'ALLOGA FRANCE', 'BETA PHARMA']  
    postcodes = ['NG', 'LU', 'NN', 'NZ', 'AK']

    Expected Output format:
    
    matched_customers: ['ALLOGA UK','ALLOGA FRANCE', 'FORUM', 'USA']
    matched_postcodes: ['NG', 'LU']
    


    Process the inputs {selected_customers}, {selected_postcodes}, {customers}, and {postcodes} and return the final answer that should contain only two lists with no explanation.

    <answer>
    matched_customers: ['ALLOGA UK','ALLOGA FRANCE', 'FORUM', 'USA']
    matched_postcodes: ['NG', 'LU']
    </answer>
    """
        
    # Call OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": "You are an assistant skilled at answering questions about searching something"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.7
    )
    return response.choices[0].message.content


def get_parameters_values(api_key, query):

    instructions = """You are an AI assistant tasked with analyzing questions and based on that give value of certain variables:
    list of variables are below:
    start_date:
    end_date:
    group_method:
    all_post_code: 
    all_customers:
    selected_postcodes: 
    selected_customers:

    I will provide you a question to answer, based on the question you need to provide variable values .

    Here are some sample questions I would like you to answer:
    1. How can I optimize the shipment costs for user ALLOGA UK.
    2. Can you optimize costs for shipments to zip code NG between January and March 2024?

    To answer this, first think through your approach
    To answer this question, 
    1. You will need to find the start and end date first if it is not mentioned then start date will be 1st january 2023 and end date will be 30th November 2024
    2. Determine the group_method, whether it 'Customer Level' or 'Post Code Level'
    3. Determine the list of post codes or list of users that are mentioned in the question, if there is no mention of post code or users , then make all_post_code = False  if group method is Post Code Level otherwise keep it None, and  all_customers = False if group method is Customer Level otherwise keep it None.
    4. if there is a mention of certain users or zip codes, make a list of that.

    return the value of all the required variables based on the questions in json format.

    for example for the first question "How can I optimize the shipment costs for user ALLOGA UK." the response should be similar to this but in dictionary format:
    
    expected output format:

    start_date: 2023-01-01
    end_date: 2024-11-30
    group_method: 'Customer Level'
    all_post_code: None
    all_customers: False
    selected_postcodes: []
    selected_customers:  [ALLOGA UK]


    for the 2nd question "Can you optimize costs for shipments to zip code NG (313) between January and March 2024?",  response should be similar to this but in dictionary format:
    
    expected output format:

   
    start_date: 2024-01-01
    end_date: 2024-01-31
    group_method: 'Post Code Level'
    all_post_code: False
    all_customers: None
    selected_postcodes: [NG]
    selected_customers:  []
   

    Note : if someone mention last month or recent month,  keep it November 2024, and date format should be: yyyy-mm-dd

    strict instructions: The final output should be only in this format (no extra text or steps should be included in the output):

    { "start_date": "2024-11-01",
    "end_date": "2024-11-30",
    "group_method": "Post Code Level",
    "all_post_code": True,
    "all_customers": None,
    "selected_postcodes": [],
    "selected_customers": [] }

    """
    response = get_chatgpt_response(api_key, instructions, query)
    print(response)
    if response:
        try:
            extracted_code= eval(response)
            input=pd.read_excel("Complete Input.xlsx")
            customers=input["NAME"].unique()
            postcodes=input["SHORT_POSTCODE"].unique()
            selected_customers= extracted_code['selected_customers']
            selected_postcodes= extracted_code['selected_postcodes']
            answer = ask_openai(selected_customers,selected_postcodes,customers,postcodes)
            
            # Extract matched_customers
            customers_match = re.search(r"matched_customers:\s*(\[.*\])", answer)
            matched_customers = ast.literal_eval(customers_match.group(1)) if customers_match else []

            # Extract matched_postcodes
            postcodes_match = re.search(r"matched_postcodes:\s*(\[.*\])", answer)
            matched_postcodes = ast.literal_eval(postcodes_match.group(1)) if postcodes_match else []

            print(matched_customers)
            print(matched_postcodes)
            extracted_code['selected_customers']= matched_customers
            extracted_code['selected_postcodes']= matched_postcodes

            return extracted_code
    
    ### return default parameters:
        except: 
            default_param={
            "start_date": "01/01/2024",
            "end_date": "31/03/2024",
            "group_method": "Post Code Level",
            "all_post_code": False,
            "all_customers": None,
            "selected_postcodes": ["NG"],
            "selected_customers": [] }

            return default_param

############################################################################################################################################################################################


def get_filtered_data(parameters, df):

    global group_field
    global group_method

    group_method = parameters['group_method']
    group_field = 'SHORT_POSTCODE' if group_method == 'Post Code Level' else 'NAME'

    # Month selection
    start_date= parameters['start_date']
    end_date= parameters['end_date']

    # Filter data based on selected date range
    df = df[(df['SHIPPED_DATE'] >= start_date) & (df['SHIPPED_DATE'] <= end_date)]
    # print("only date filter", df.shape) ### checkk

    # Add checkbox and conditional dropdown for selecting post codes or customers
  
    if group_method == 'Post Code Level':
        all_postcodes = parameters['all_post_code']
        
        if not all_postcodes:
            selected_postcodes = parameters['selected_postcodes']
            selected_postcodes= [z.strip('') for z in selected_postcodes ]
    else:  # Customer Level
        all_customers = parameters['all_customers']
        if not all_customers:
            selected_customers = parameters['selected_customers']
            selected_customers= [c.strip('') for c in selected_customers]
    # Filter the dataframe based on the selection
    if group_method == 'Post Code Level' and not all_postcodes:
        if selected_postcodes:  # Only filter if some postcodes are selected
            df = df[df['SHORT_POSTCODE'].str.strip('').isin(selected_postcodes)]
        else:
            return pd.DataFrame()
        
    elif group_method == 'Customer Level' and not all_customers:
        if selected_customers:  # Only filter if some customers are selected
            df = df[df['NAME'].str.strip('').isin(selected_customers)]
        else :
            return pd.DataFrame()
        
    return df



def calculate_metrics(all_consolidated_shipments, df):
    total_orders = sum(len(shipment['Orders']) for shipment in all_consolidated_shipments)
    total_shipments = len(all_consolidated_shipments)
    total_pallets = sum(shipment['Total Pallets'] for shipment in all_consolidated_shipments)
    total_utilization = sum(shipment['Utilization %'] for shipment in all_consolidated_shipments)
    average_utilization = total_utilization / total_shipments if total_shipments > 0 else 0
    total_shipment_cost = sum(shipment['Shipment Cost'] for shipment in all_consolidated_shipments if not pd.isna(shipment['Shipment Cost']))
    total_baseline_cost = sum(shipment['Baseline Cost'] for shipment in all_consolidated_shipments if not pd.isna(shipment['Baseline Cost']))
    cost_savings = total_baseline_cost - total_shipment_cost
    percent_savings = (cost_savings / total_baseline_cost) * 100 if total_baseline_cost > 0 else 0

    # Calculate CO2 Emission
    total_distance = 0
    sum_dist = 0
    for shipment in all_consolidated_shipments:
        order_ids = shipment['Orders']
        avg_distance = df[df['ORDER_ID'].isin(order_ids)]['Distance'].mean()
        sum_distance = df[df['ORDER_ID'].isin(order_ids)]['Distance'].sum()
        total_distance += avg_distance
        sum_dist += sum_distance
    co2_emission = (sum_dist - total_distance) * 2  # Multiply by 2 


    metrics = {
        'Total Orders': total_orders,
        'Total Shipments': total_shipments,
        'Total Pallets': total_pallets,
        'Average Utilization': average_utilization,
        'Total Shipment Cost': total_shipment_cost,
        'Total Baseline Cost': total_baseline_cost,
        'Cost Savings': round(cost_savings,1),
        'Percent Savings': percent_savings,
        'CO2 Emission': co2_emission
    }

    return metrics



def analyze_consolidation_distribution(all_consolidated_shipments, df):
    distribution = {}
    for shipment in all_consolidated_shipments:
        consolidation_date = shipment['Date']
        for order_id in shipment['Orders']:
            shipped_date = df.loc[df['ORDER_ID'] == order_id, 'SHIPPED_DATE'].iloc[0]
            days_difference = (shipped_date - consolidation_date).days
            if days_difference not in distribution:
                distribution[days_difference] = 0
            distribution[days_difference] += 1
    
    total_orders = sum(distribution.values())
    distribution_percentage = {k: round((v / total_orders) * 100, 1) for k, v in distribution.items()}
    return distribution, distribution_percentage



def create_utilization_chart(all_consolidated_shipments):
    print(all_consolidated_shipments)  #### checkkkk
    utilization_bins = {f"{i}-{i+5}%": 0 for i in range(0, 100, 5)}
    for shipment in all_consolidated_shipments:
        utilization = shipment['Utilization %']
        bin_index = min(int(utilization // 5) * 5, 95)  # Cap at 95-100% bin
        bin_key = f"{bin_index}-{bin_index+5}%"
        utilization_bins[bin_key] += 1

    total_shipments = len(all_consolidated_shipments)
    utilization_distribution = {bin: (count / total_shipments) * 100 for bin, count in utilization_bins.items()}

    fig = go.Figure(data=[go.Bar(x=list(utilization_distribution.keys()), y=list(utilization_distribution.values()), marker_color='#1f77b4')])
    fig.update_layout(
        title={'text': 'Utilization Distribution', 'font': {'size': 22}},
        xaxis_title='Utilization Range',
        yaxis_title='Percentage of Shipments', 
        width=1000, 
        height=500
        )

    return fig


############################################################################################################################################################################################

def calculate_priority(shipped_date, current_date, shipment_window):
    days_left = (shipped_date - current_date).days
    if 0 <= days_left <= shipment_window:
        return days_left
    return np.nan

def best_fit_decreasing(items, capacity):
    items = sorted(items, key=lambda x: x['Total Pallets'], reverse=True)
    shipments = []

    for item in items:
        best_shipment = None
        min_space = capacity + 1

        for shipment in shipments:
            current_load = sum(order['Total Pallets'] for order in shipment)
            new_load = current_load + item['Total Pallets']
            
            if new_load <= capacity:
                space_left = capacity - current_load
            else:
                continue  # Skip this shipment if adding the item would exceed capacity
            
            if item['Total Pallets'] <= space_left < min_space:
                best_shipment = shipment
                min_space = space_left

        if best_shipment:
            best_shipment.append(item)
        else:
            shipments.append([item])

    return shipments

def get_baseline_cost(prod_type, short_postcode, pallets,rate_card):
    total_cost = 0
    for pallet in pallets:
        cost = get_shipment_cost(prod_type, short_postcode, pallet,rate_card)
        if pd.isna(cost):
            return np.nan
        total_cost += cost
    return round(total_cost, 1)


############################################################################################################################################################################################

def load_data():
    rate_card_ambient = pd.read_excel('Complete Input.xlsx', sheet_name='AMBIENT')
    rate_card_ambcontrol = pd.read_excel('Complete Input.xlsx', sheet_name='AMBCONTROL')
    return rate_card_ambient, rate_card_ambcontrol

def get_shipment_cost(prod_type, short_postcode, total_pallets,rate_card):
    rate_card_ambient,rate_card_ambcontrol = rate_card["rate_card_ambient"],rate_card["rate_card_ambcontrol"]
    if prod_type == 'AMBIENT':
        rate_card = rate_card_ambient
    elif prod_type == 'AMBCONTROL':
        rate_card = rate_card_ambcontrol
    else:
        return np.nan

    row = rate_card[rate_card['SHORT_POSTCODE'] == short_postcode]
    
    if row.empty:
        return np.nan

    cost_per_pallet = row.get(total_pallets, np.nan).values[0]

    if pd.isna(cost_per_pallet):
        return np.nan

    shipment_cost = round(cost_per_pallet * total_pallets, 1)
    return shipment_cost


def process_shipment(shipment, consolidated_shipments, allocation_matrix, working_df, current_date, capacity):
    total_pallets = sum(order['Total Pallets'] for order in shipment)
    utilization = (total_pallets / capacity) * 100

    prod_type = shipment[0]['PROD TYPE']
    short_postcode = shipment[0]['SHORT_POSTCODE']
########################################################################################################################
    shipment_cost = get_shipment_cost(prod_type, short_postcode, total_pallets,rate_card)
########################################################################################################################

    pallets = [order['Total Pallets'] for order in shipment]
########################################################################################################################
    baseline_cost = get_baseline_cost(prod_type, short_postcode, pallets,rate_card)
########################################################################################################################
    shipment_info = {
        'Date': current_date,
        'Orders': [order['ORDER_ID'] for order in shipment],
        'Total Pallets': total_pallets,
        'Capacity': capacity,
        'Utilization %': round(utilization, 1),
        'Order Count': len(shipment),
        'Pallets': pallets,
        'PROD TYPE': prod_type,
        'GROUP': shipment[0]['GROUP'],
        'Shipment Cost': shipment_cost,
        'Baseline Cost': baseline_cost,
        'SHORT_POSTCODE': short_postcode,
        'Load Type': 'Full' if total_pallets > 26 else 'Partial'
    }

    if group_method == 'NAME':
        shipment_info['NAME'] = shipment[0]['NAME']

    consolidated_shipments.append(shipment_info)
    
    for order in shipment:
        allocation_matrix.loc[order['ORDER_ID'], current_date] = 1
        working_df.drop(working_df[working_df['ORDER_ID'] == order['ORDER_ID']].index, inplace=True)


def consolidate_shipments(df, high_priority_limit, utilization_threshold, shipment_window, date_range, progress_callback, capacity,rate_card):
    consolidated_shipments = []
    allocation_matrix = pd.DataFrame(0, index=df['ORDER_ID'], columns=date_range)
    
    working_df = df.copy()
    
    for current_date in date_range:
########################################################################################################################
        working_df.loc[:, 'Priority'] = working_df['SHIPPED_DATE'].apply(lambda x: calculate_priority(x, current_date, shipment_window))
########################################################################################################################

        if (working_df['Priority'] == 0).any():
            eligible_orders = working_df[working_df['Priority'].notnull()].sort_values('Priority')
            high_priority_orders = eligible_orders[eligible_orders['Priority'] <= high_priority_limit].to_dict('records')
            low_priority_orders = eligible_orders[eligible_orders['Priority'] > high_priority_limit].to_dict('records')
            
            if high_priority_orders or low_priority_orders:
########################################################################################################################                
                # Process high priority orders first
                high_priority_shipments = best_fit_decreasing(high_priority_orders, capacity)
########################################################################################################################

                # Try to fill high priority shipments with low priority orders
                for shipment in high_priority_shipments:
                    current_load = sum(order['Total Pallets'] for order in shipment)
                    space_left = capacity - current_load  # Use the variable capacity
                    
                    if space_left > 0:
                        low_priority_orders.sort(key=lambda x: x['Total Pallets'], reverse=True)
                        for low_priority_order in low_priority_orders[:]:
                            if low_priority_order['Total Pallets'] <= space_left:
                                shipment.append(low_priority_order)
                                space_left -= low_priority_order['Total Pallets']
                                low_priority_orders.remove(low_priority_order)
                            if space_left == 0:
                                break
                
                # Process remaining low priority orders
                low_priority_shipments = best_fit_decreasing(low_priority_orders, capacity)
                
                # Process all shipments
                all_shipments = high_priority_shipments + low_priority_shipments
                for shipment in all_shipments:
                    total_pallets = sum(order['Total Pallets'] for order in shipment)
                    utilization = (total_pallets / capacity) * 100
                    
                    # Always process shipments with high priority orders, apply threshold only to pure low priority shipments
                    if any(order['Priority'] <= high_priority_limit for order in shipment) or utilization >= utilization_threshold:
########################################################################################################################
                        process_shipment(shipment, consolidated_shipments, allocation_matrix, working_df, current_date, capacity,rate_card)
########################################################################################################################        
        progress_callback()
    
    return consolidated_shipments, allocation_matrix




############################################################################################################################################################################################
