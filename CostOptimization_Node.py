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
import streamlit as st
import matplotlib.pyplot as plt
from IPython.display import display, Markdown,Image as IPythonImage
from PIL import Image

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
from langchain_core.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent


from config import llm, display_saved_plot
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo",api_key = openai.api_key,temperature=0)


from cost_optimization import (
    llm, get_parameters_values, consolidate_shipments, calculate_metrics,
    analyze_consolidation_distribution, get_filtered_data
)



# Defining Agent Helper Functions....#######################################################################################################################

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

# Defining Plotly-Bokeh Helper Functions....#######################################################################################################################


def create_consolidated_shipments_calendar(consolidated_df):
    # Group by Date and calculate both Shipments Count and Total Orders
    df_consolidated = consolidated_df.groupby('Date').agg({
        'Orders': ['count', lambda x: sum(len(orders) for orders in x)]
    }).reset_index()
    df_consolidated.columns = ['Date', 'Shipments Count', 'Orders Count']
    
    # Split data by year
    df_2023 = df_consolidated[df_consolidated['Date'].dt.year == 2023]
    df_2024 = df_consolidated[df_consolidated['Date'].dt.year == 2024]
    
    calendar_data_2023 = df_2023[['Date', 'Shipments Count', 'Orders Count']].values.tolist()
    calendar_data_2024 = df_2024[['Date', 'Shipments Count', 'Orders Count']].values.tolist()

    def create_calendar(data, year):
        return (
            Calendar(init_opts=opts.InitOpts(width="984px", height="200px", theme=ThemeType.ROMANTIC))
            .add(
                series_name="",
                yaxis_data=data,
                calendar_opts=opts.CalendarOpts(
                    pos_top="50",
                    pos_left="40",
                    pos_right="30",
                    range_=str(year),
                    yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
                    daylabel_opts=opts.CalendarDayLabelOpts(name_map=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']),
                    monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="en"),
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"Calendar Heatmap for Orders and Shipments After Consolidation ({year})"),
                visualmap_opts=opts.VisualMapOpts(
                    max_=max(item[2] for item in data) if data else 0,
                    min_=min(item[2] for item in data) if data else 0,
                    orient="horizontal",
                    is_piecewise=False,
                    pos_bottom="20",
                    pos_left="center",
                    range_color=["#E8F5E9", "#1B5E20"],
                    is_show=False,
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        """
                        function (p) {
                            var date = new Date(p.data[0]);
                            var day = date.getDate().toString().padStart(2, '0');
                            var month = (date.getMonth() + 1).toString().padStart(2, '0');
                            var year = date.getFullYear();
                            return 'Date: ' + day + '/' + month + '/' + year + 
                                   '<br/>Orders: ' + p.data[2] +
                                   '<br/>Shipments: ' + p.data[1];
                        }
                        """
                    )
                )
            )
        )

    calendar_2023 = create_calendar(calendar_data_2023, 2023)
    calendar_2024 = create_calendar(calendar_data_2024, 2024)

    return calendar_2023, calendar_2024






def create_original_orders_calendar(original_df):
    df_original = original_df.groupby('SHIPPED_DATE').size().reset_index(name='Orders Shipped')
    
    # Split data by year
    df_2023 = df_original[df_original['SHIPPED_DATE'].dt.year == 2023]
    df_2024 = df_original[df_original['SHIPPED_DATE'].dt.year == 2024]
    
    calendar_data_2023 = df_2023[['SHIPPED_DATE', 'Orders Shipped']].values.tolist()
    calendar_data_2024 = df_2024[['SHIPPED_DATE', 'Orders Shipped']].values.tolist()

    def create_calendar(data, year):
        return (
            Calendar(init_opts=opts.InitOpts(width="984px", height="200px", theme=ThemeType.ROMANTIC))
            .add(
                series_name="",
                yaxis_data=data,
                calendar_opts=opts.CalendarOpts(
                    pos_top="50",
                    pos_left="40",
                    pos_right="30",
                    range_=str(year),
                    yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
                    daylabel_opts=opts.CalendarDayLabelOpts(name_map=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']),
                    monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="en"),
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"Calendar Heatmap for Orders Shipped Before Consolidation ({year})"),
                visualmap_opts=opts.VisualMapOpts(
                    max_=max(item[1] for item in data) if data else 0,
                    min_=min(item[1] for item in data) if data else 0,
                    orient="horizontal",
                    is_piecewise=False,
                    pos_bottom="20",
                    pos_left="center",
                    range_color=["#E8F5E9", "#1B5E20"],
                    is_show=False,
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        """
                        function (p) {
                            var date = new Date(p.data[0]);
                            var day = date.getDate().toString().padStart(2, '0');
                            var month = (date.getMonth() + 1).toString().padStart(2, '0');
                            var year = date.getFullYear();
                            return 'Date: ' + day + '/' + month + '/' + year + '<br/>Orders: ' + p.data[1];
                        }
                        """
                    )
                )
            )
        )

    calendar_2023 = create_calendar(calendar_data_2023, 2023)
    calendar_2024 = create_calendar(calendar_data_2024, 2024)

    return calendar_2023, calendar_2024

def create_heatmap_and_bar_charts(consolidated_df, original_df, start_date, end_date):
    # Create calendar charts (existing code)
    chart_original_2023, chart_original_2024 = create_original_orders_calendar(original_df)
    chart_consolidated_2023, chart_consolidated_2024 = create_consolidated_shipments_calendar(consolidated_df)
    
    # Create bar charts for orders over time
    def create_bar_charts(df_original, df_consolidated, year):
        # Filter data for the specific year
        mask_original = df_original['SHIPPED_DATE'].dt.year == year
        year_data_original = df_original[mask_original]
        
        # For consolidated data
        if 'Date' in df_consolidated.columns:
            mask_consolidated = pd.to_datetime(df_consolidated['Date']).dt.year == year
            year_data_consolidated = df_consolidated[mask_consolidated]
        else:
            year_data_consolidated = pd.DataFrame()
        
        # Create subplot figure with shared x-axis
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f'Daily Orders Before Consolidation ({year})',
                f'Daily Orders After Consolidation ({year})'
            )
        )
        
        # Add bar chart for original orders
        if not year_data_original.empty:
            daily_orders = year_data_original.groupby('SHIPPED_DATE').size().reset_index()
            daily_orders.columns = ['Date', 'Orders']
            
            fig.add_trace(
                go.Bar(
                    x=daily_orders['Date'],
                    y=daily_orders['Orders'],
                    name='Orders',
                    marker_color='#1f77b4'
                ),
                row=1, 
                col=1
            )
        
        # Add bar chart for consolidated orders
        if not year_data_consolidated.empty:
            daily_consolidated = year_data_consolidated.groupby('Date').agg({
                'Orders': lambda x: sum(len(orders) for orders in x)
            }).reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=daily_consolidated['Date'],
                    y=daily_consolidated['Orders'],
                    name='Orders',
                    marker_color='#749f77'
                ),
                row=2, 
                col=1
            )
        
        # Update layout
        fig.update_layout(
            height=500,  # Increased height for better visibility
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=20, t=60, b=20),
            hovermode='x unified'
        )
        
        # Update x-axes
        fig.update_xaxes(
            rangeslider=dict(
                visible=True,
                thickness=0.05,  # Make the rangeslider thinner
                bgcolor='#F4F4F4',  # Light gray background
                bordercolor='#DEDEDE',  # Slightly darker border
            ),
            row=2,
            col=1
        )
        fig.update_xaxes(
            rangeslider=dict(visible=False),
            row=1,
            col=1
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Number of Orders", row=1, col=1)
        fig.update_yaxes(title_text="Number of Orders", row=2, col=1)
        
        return fig
    
    # Create bar charts for both years
    bar_charts_2023 = create_bar_charts(original_df, consolidated_df, 2023)
    bar_charts_2024 = create_bar_charts(original_df, consolidated_df, 2024)
    
    return {
        2023: (chart_original_2023, chart_consolidated_2023, bar_charts_2023),
        2024: (chart_original_2024, chart_consolidated_2024, bar_charts_2024)
    }



def create_shipment_window_vs_saving_plot(all_results,best_params):
    # Create a dataframe with all simulation results
    results_df = pd.DataFrame(all_results)

    
    # Preprocess the data to keep only the row with max Cost Savings for each Shipment Window
    optimal_results = results_df.loc[results_df.groupby(['Shipment Window'])['Cost Savings'].idxmax()]
    
    # Create ColumnDataSource
    source = ColumnDataSource(optimal_results)

    
    # Display the Shipment Window Comparison chart
    st.markdown("<h2 style='font-size:24px;'>Shipment Window Comparison</h2>", unsafe_allow_html=True)

    shipment_text = (
        f"For each shipment window:\n\n"
        f"- Shipments are grouped together through the consolidation function.\n"
        f"- Key performance metrics, such as cost savings, utilization, and emissions, are calculated.\n"
        f"- The cost savings are compared across different shipment windows to identify the most efficient one.\n"
        f"- On analyzing this data , the best shipment window is observed to be  **{best_params[0]}** days."
    )
    # shipment_rephrase_text = rephrase_text(api_key , shipment_text)
    st.write(shipment_text)

    # Select the best rows for each shipment window
    best_results = results_df.loc[results_df.groupby('Shipment Window')['Percent Savings'].idxmax()]
    
    # Sort by Shipment Window
    best_results = best_results.sort_values('Shipment Window')
    
    # Create a complete range of shipment windows from 0 to 30
    all_windows = list(range(0, 31))
    
    # Create the subplot figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add the stacked bar chart
    fig.add_trace(
        go.Bar(
            x=all_windows,
            y=[best_results[best_results['Shipment Window'] == w]['Total Shipment Cost'].values[0] if w in best_results['Shipment Window'].values else 0 for w in all_windows],
            name='Total Shipment Cost',
            marker_color='#1f77b4'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=all_windows,
            y=[best_results[best_results['Shipment Window'] == w]['Cost Savings'].values[0] if w in best_results['Shipment Window'].values else 0 for w in all_windows],
            name='Cost Savings',
            marker_color='#a9d6a9'
        )
    )
    
    # Add the line chart for Total Shipments on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=all_windows,
            y=[best_results[best_results['Shipment Window'] == w]['Total Shipments'].values[0] if w in best_results['Shipment Window'].values else None for w in all_windows],
            name='Total Shipments',
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8),
            hovertemplate='<b>Shipment Window</b>: %{x}<br>' +
                            '<b>Total Shipments</b>: %{y}<br>' +
                            '<b>Average Utilization</b>: %{text:.1f}%<extra></extra>',
            text=[best_results[best_results['Shipment Window'] == w]['Average Utilization'].values[0] if w in best_results['Shipment Window'].values else None for w in all_windows],
        ),
        secondary_y=True
    )
    
    # Add text annotations for Percent Savings
    for w in all_windows:
        if w in best_results['Shipment Window'].values:
            row = best_results[best_results['Shipment Window'] == w].iloc[0]
            fig.add_annotation(
                x=w,
                y=row['Total Shipment Cost'] + row['Cost Savings'],
                text=f"{row['Percent Savings']:.1f}%",
                showarrow=False,
                yanchor='bottom',
                yshift=5,
                font=dict(size=10)
            )
    
    # Update the layout
    fig.update_layout(
        barmode='stack',
        height=600,
        width=1050,
        # margin=dict(l=50, r=50, t=40, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.update_xaxes(title_text='Shipment Window', tickmode='linear', dtick=1, range=[-0.5, 30.5])
    fig.update_yaxes(title_text='Cost (£)', secondary_y=False)
    fig.update_yaxes(title_text='Total Shipments', secondary_y=True)
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=False)




def create_calendar_heatmap_before_vs_after(parameters):

    charts = create_heatmap_and_bar_charts(parameters['all_consolidated_shipments'], parameters['filtered_df'], parameters['start_date'], parameters['end_date'])

    years_in_range = set(pd.date_range(parameters['start_date'], parameters['end_date']).year)

    with st.expander("Heatmap Analysis Charts(Before & After Consolidation)"):
        for year in [2023, 2024]:
            if year in years_in_range:
                chart_original, chart_consolidated, bar_comparison = charts[year]

                # Display heatmaps for the current year
                st.write(f"**Heatmaps for the year {year} (Before & After Consolidation):**")
                st.components.v1.html(chart_original.render_embed(), height=216, width=1000)
                st.components.v1.html(chart_consolidated.render_embed(), height=216, width=1000)

                # Add a divider between years
                if year == 2023:
                    st.write("----------------------")  # Optional divider for visual separation

        # After the loop, you can add the interpretation section just once
        st.write("""
                    **Heatmap Interpretation:**

                    - **Dark Green Areas**: Indicate high shipment concentration on specific dates, showcasing peak activity where most orders are processed.
                    - **Lighter Green Areas**: Represent fewer or no shipments, highlighting potential inefficiencies in the initial shipment strategy before optimization.

                    **Before Consolidation:**

                    - Shipments were frequent but scattered across multiple days without strategic grouping.
                    - Increased costs due to multiple underutilized shipments.
                    - Truck utilization remained suboptimal, leading to excess operational expenses.

                    **After Consolidation:**

                    - Orders were intelligently grouped into fewer shipments, reducing the total number of trips while maintaining service levels.
                    - Optimized cost savings through better utilization and fewer underfilled shipments.
                    - Enhanced planning efficiency, enabling better decision-making for future shipment scheduling.
                    """)


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
        self.utilization_threshold = 95

    def load_data(self):
        self.rate_card_ambient = pd.read_excel('Complete Input.xlsx',sheet_name="AMBIENT")
        self.rate_card_ambcontrol = pd.read_excel('Complete Input.xlsx',sheet_name="AMBCONTROL")
        return {"rate_card_ambient":self.rate_card_ambient,"rate_card_ambcontrol":self.rate_card_ambcontrol}
    
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
        group_field = 'SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME'
        st.write("Shape of original data after filtering:",df.shape)
        
        df['GROUP'] = df['SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME']
        grouped = df.groupby(['PROD TYPE', 'GROUP'])
        date_range = pd.date_range(start=self.parameters['start_date'], end=self.parameters['end_date'])

        
        best_metrics=None
        best_consolidated_shipments=None
        best_params = None

        all_results = []
        rate_card = self.load_data()
        for shipment_window in range(self.shipment_window_range[0], self.shipment_window_range[1] + 1):
            # st.write(f"Consolidating orders for shipment window: {shipment_window}")
            st.toast(f"Consolidating orders for shipment window: {shipment_window}")
            high_priority_limit = 0
            all_consolidated_shipments = []
            for _, group_df in grouped:
                consolidated_shipments, _ = consolidate_shipments(
                    group_df, high_priority_limit, self.utilization_threshold, shipment_window, date_range, lambda: None, self.total_shipment_capacity,rate_card
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

        # Update best results if current combination is better
        if best_metrics is None or metrics['Cost Savings'] > best_metrics['Cost Savings']:
            best_metrics = metrics
            best_consolidated_shipments = all_consolidated_shipments
            best_params = (shipment_window, high_priority_limit, self.utilization_threshold)


        summary_text = (
            f"Optimizing outbound deliveries and identifying cost-saving opportunities involve analyzing various factors "
            f"such as order patterns, delivery routes, shipping costs, and consolidation opportunities.\n\n"

            f"On analyzing the data, I can provide some estimates of cost savings on the historical data if we were to "
            f"group orders to consolidate deliveries.\n\n"

            "**APPROACH TAKEN**\n\n"  # Ensure it's already in uppercase and same.
            f"To consolidate the deliveries, A heuristic approach was used, and the methodology is as follows:\n\n"

            f"**Group Shipments**: Orders are consolidated within a shipment window to reduce transportation costs while "
            f"maintaining timely deliveries. A shipment window represents the number of days prior to the current delivery "
            f"that the order could be potentially shipped, thus representing an opportunity to group it with earlier deliveries.\n\n"

            f"**Iterate Over Shipment Windows**: The model systematically evaluates all possible shipment windows, testing "
            f"different configurations to identify the most effective scheduling approach.\n\n"

            f"**Performance Metric Calculation**: Key performance metrics are assessed for each shipment window, including:\n"
            f"- **Cost savings**\n"
            f"- **Utilization rate**\n"
            f"- **CO2 emission reduction**\n\n"

            f"**Comparison and Selection**: After evaluating all configurations, the shipment window that maximizes cost savings "
            f"while maintaining operational efficiency is identified, and results are displayed as per the best parameter.\n\n"

            f"This method allows us to optimize logistics operations dynamically, ensuring that both financial and environmental "
            f"factors are balanced effectively."
            )


        st.write(summary_text)


        # Updating the parameters with adding shipment window vs cost saving table..    
        self.parameters['all_results'] = pd.DataFrame(all_results)
        self.parameters['best_params'] = best_params

############################################################################################################################################

        # calc_shipment_window = best_params[0]
        # calc_utilization_threshold  = self.utilization_threshold

        # if True:
        #     # start_time = time.time()

        #     with st.spinner("Calculating..."):
        #         # Prepare data for calculation
        #         df['GROUP'] = df[group_field]
        #         grouped = df.groupby(['PROD TYPE', 'GROUP'])
        #         date_range = pd.date_range(start=self.parameters['start_date'], end=self.parameters['end_date'])

        #         calc_high_priority_limit = 0
        #         all_consolidated_shipments = []
        #         all_allocation_matrices = []

        #         # Run calculation
        #         #progress_bar = custom_progress_bar()

        #         for i, ((prod_type, group), group_df) in enumerate(grouped):
        #             consolidated_shipments, allocation_matrix = consolidate_shipments(
        #                 group_df, calc_high_priority_limit, calc_utilization_threshold,
        #                 calc_shipment_window, date_range, lambda: None, total_shipment_capacity
        #             )
        #             all_consolidated_shipments.extend(consolidated_shipments)
        #             all_allocation_matrices.append(allocation_matrix)
        #             #progress_percentage = int(((i + 1) / len(grouped)) * 100)
        #             #progress_bar(progress_percentage)

        #         selected_postcodes = ", ".join(self.parameters["selected_postcodes"]) if self.parameters["selected_postcodes"] else "All Postcodes"
        #         selected_customers = ", ".join(self.parameters["selected_customers"]) if self.parameters["selected_customers"] else "All Customers"

        #         metrics = calculate_metrics(all_consolidated_shipments, df)
        #         st.markdown("<h2 style='font-size:24px;'>Identified cost savings and Key Performance Indicators (KPIs)</h2>", unsafe_allow_html=True)
        #         main_text = (
        #             f"Through extensive analysis, the OPTIMAL SHIPMENT WINDOW was determined to be **{best_params[0]}**, "
        #             f"with a PALLET SIZE of **46** for **postcodes**: {selected_postcodes} and **customers**: {selected_customers}."
        #             f"These optimizations resulted in SIGNIFICANT EFFICIENCY IMPROVEMENTS:\n\n"

        #             f"**SHIPMENT WINDOW**: The most effective shipment window was identified as ****{best_params[0]} DAYS**.\n\n"

        #             f"**COST SAVINGS**: A reduction of **£{metrics['Cost Savings']:,.1f}**, equating to an **£{metrics['Percent Savings']:.1f}%** decrease in overall transportation costs.\n\n"

        #             f"**ORDER & SHIPMENT SUMMARY**:\n"
        #             f"- TOTAL ORDERS PROCESSED: **{metrics['Total Orders']:,}** \n"
        #             f"- TOTAL SHIPMENTS MADE: **{metrics['Total Shipments']:,}**\n\n"

        #             f"**UTILIZATION EFFICIENCY**:\n"
        #             f"- AVERAGE TRUCK UTILIZATION increased to **{metrics['Average Utilization']:.1f}%**, ensuring fewer trucks operate at low capacity.\n\n"

        #             f"**ENVIRONMENTAL IMPACT**:\n"
        #             f"- CO2 EMISSIONS REDUCTION: A decrease of **{metrics['CO2 Emission']:,.1f} Kg**, supporting sustainability efforts and reducing the carbon footprint.\n\n"

        #             f"These optimizations not only lead to substantial COST REDUCTIONS but also enhance OPERATIONAL SUSTAINABILITY, "
        #             f"allowing logistics operations to function more efficiently while MINIMIZING ENVIRONMENTAL IMPACT."
        #         )

        #         # rephrase_main_text = rephrase_text(api_key, main_text)
        #         st.write(main_text)



    
    def consolidate_for_shipment_window(self):
        """Runs consolidation algorithm based on the selected shipment window."""
        df = self.get_filtered_df_from_question()
        df['GROUP'] = df['SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME']
        grouped = df.groupby(['PROD TYPE', 'GROUP'])
        date_range = pd.date_range(start=self.parameters['start_date'], end=self.parameters['end_date'])
        
        all_consolidated_shipments = []
        rate_card = self.load_data()
        for _, group_df in grouped:
            consolidated_shipments, _ = consolidate_shipments(
                group_df, 0, 95, self.parameters['window'], date_range, lambda: None, self.total_shipment_capacity,rate_card
            )
            all_consolidated_shipments.extend(consolidated_shipments)

        
        selected_postcodes = ", ".join(self.parameters["selected_postcodes"]) if self.parameters["selected_postcodes"] else "All Postcodes"
        selected_customers = ", ".join(self.parameters["selected_customers"]) if self.parameters["selected_customers"] else "All Customers"

        metrics = calculate_metrics(all_consolidated_shipments, df)
        st.markdown("<h2 style='font-size:24px;'>Identified cost savings and Key Performance Indicators (KPIs)</h2>", unsafe_allow_html=True)
        main_text = (
            f"Through extensive analysis, the OPTIMAL SHIPMENT WINDOW was determined to be **{self.parameters['best_params'][0]}**, "
            f"with a PALLET SIZE of **46** for **postcodes**: {selected_postcodes} and **customers**: {selected_customers}."
            f"These optimizations resulted in SIGNIFICANT EFFICIENCY IMPROVEMENTS:\n\n"

            f"**SHIPMENT WINDOW**: The most effective shipment window was identified as **{self.parameters['best_params'][0]} DAYS**.\n\n"

            f"**COST SAVINGS**: A reduction of **£{metrics['Cost Savings']:,.1f}**, equating to an **£{metrics['Percent Savings']:.1f}%** decrease in overall transportation costs.\n\n"

            f"**ORDER & SHIPMENT SUMMARY**:\n"
            f"- TOTAL ORDERS PROCESSED: **{metrics['Total Orders']:,}** \n"
            f"- TOTAL SHIPMENTS MADE: **{metrics['Total Shipments']:,}**\n\n"

            f"**UTILIZATION EFFICIENCY**:\n"
            f"- AVERAGE TRUCK UTILIZATION increased to **{metrics['Average Utilization']:.1f}%**, ensuring fewer trucks operate at low capacity.\n\n"

            f"**ENVIRONMENTAL IMPACT**:\n"
            f"- CO2 EMISSIONS REDUCTION: A decrease of **{metrics['CO2 Emission']:,.1f} Kg**, supporting sustainability efforts and reducing the carbon footprint.\n\n"

            f"These optimizations not only lead to substantial COST REDUCTIONS but also enhance OPERATIONAL SUSTAINABILITY, "
            f"allowing logistics operations to function more efficiently while MINIMIZING ENVIRONMENTAL IMPACT."
        )

        # rephrase_main_text = rephrase_text(api_key, main_text)
        st.write(main_text)

        
        self.parameters['all_consolidated_shipments'] = pd.DataFrame(all_consolidated_shipments)
        self.parameters['filtered_df'] = df

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
        
        comparison_df_dict = comparison_df.to_dict()



        # Create three columns for before, after, and change metrics
        col1,col2,col3 = st.columns(3)

        # Style for metric display
        metric_style = """
                            <div style="
                                background-color: #f0f2f6;
                                padding: 0px;
                                border-radius: 5px;
                                margin: 5px 0;
                            ">
                                <span style="font-weight: bold;">{label}:</span> {value}
                            </div>
                        """

        # Style for percentage changes
        change_style = """
                            <div style="
                                background-color: #e8f0fe;
                                padding: 0px;
                                border-radius: 5px;
                                margin: 5px 0;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            ">
                                <span style="font-weight: bold;">{label}:</span>
                                <span style="color: {color}; font-weight: bold;">{value:+.1f}%</span>
                            </div>
                        """
        
        # Before consolidation metrics
        with col1:
            st.markdown("##### Before Consolidation")
            st.markdown(metric_style.format(
                label="Days Shipped",
                value=f"{comparison_df_dict['Before']['Days']:,}"
            ), unsafe_allow_html=True)
            st.markdown(metric_style.format(
                label="Pallets Shipped per Day",
                value=f"{comparison_df_dict['Before']['Pallets Per Day']:.1f}"
            ), unsafe_allow_html=True)
            st.markdown(metric_style.format(
                label="Pallets per Shipment",
                value=f"{comparison_df_dict['Before']['Pallets per Shipment']:.1f}"
            ), unsafe_allow_html=True)


        # After consolidation metrics
        with col2:
            st.markdown("##### After Consolidation")
            st.markdown(metric_style.format(
                label="Days Shipped",
                value=f"{comparison_df_dict['After']['Days']:,}"
            ), unsafe_allow_html=True)
            st.markdown(metric_style.format(
                label="Pallets Shipped per Day",
                value=f"{comparison_df_dict['After']['Pallets Per Day']:.1f}"
            ), unsafe_allow_html=True)
            st.markdown(metric_style.format(
                label="Pallets per Shipment",
                value=f"{comparison_df_dict['After']['Pallets per Shipment']:.1f}"
            ), unsafe_allow_html=True)

        # Percentage changes
        with col3:
            st.markdown("##### Percentage Change")
            st.markdown(change_style.format(
                label="Days Shipped",
                value=comparison_df_dict['% Change']['Days'],
                color="blue" if comparison_df_dict['% Change']['Days'] > 0 else "green"
            ), unsafe_allow_html=True)
            st.markdown(change_style.format(
                label="Pallets Shipped per Day",
                value= comparison_df_dict['% Change']['Pallets Per Day'],
                color="green" if comparison_df_dict['% Change']['Pallets Per Day'] > 0 else "red"
            ), unsafe_allow_html=True)
            st.markdown(change_style.format(
                label="Pallets per Shipment",
                value= comparison_df_dict['% Change']['Pallets per Shipment'],
                color="green" if comparison_df_dict['% Change']['Pallets per Shipment'] > 0 else "red"
            ), unsafe_allow_html=True)

        #     st.text("")



        return comparison_df
    


    
    def handle_query(self, question):
        """Handles user queries dynamically with conversation history and data processing."""
        
        def run_agent_query(agent, query, dataframe, max_attempts=3):
            """Runs an agent query with up to `max_attempts` retries on failure.

            Args:
                agent: The agent to invoke.
                query (str): The query to pass to the agent.
                dataframe (pd.DataFrame): DataFrame for response context.
                max_attempts (int, optional): Maximum retry attempts. Defaults to 3.

            Returns:
                str: Final answer or error message after attempts.
            """
            attempt = 0
            while attempt < max_attempts:
                try:
                    # st.info(f"Attempt {attempt + 1} of {max_attempts}...")
                    response = agent.invoke(query)
                    response_ = agent_wrapper(response, dataframe)
                    
                    display_agent_steps(response_['steps'])
                    # st.success("Query processed successfully.")
                    st.write(response_["final_answer"])

                    return response_["final_answer"]

                except Exception as e:
                    attempt += 1
                    st.warning(f"Error on attempt {attempt}: {e}")

                    if attempt == max_attempts:
                        st.error(f"Failed after {max_attempts} attempts. Please revise the query or check the data.")
                        return f"Error: {e}"

        def display_agent_steps(steps):
            """Displays agent reasoning steps and associated plots."""
            for i, step in enumerate(steps):
                # st.subheader(f"Step {i + 1}")
                st.write(step['message'])
                for plot_path in step['plot_paths']:
                    display_saved_plot(plot_path)

        chat_history = [{"Human": question}]

        # Extract parameters from question
        st.info("Extracting parameters from question...")
        extracted_params = get_parameters_values(self.parameters["api_key"], question)
        self.parameters.update(extracted_params)
        chat_history.append({"Agent": f"Parameters extracted: {extracted_params}"})

        # Run cost-saving algorithm
        st.info("Running cost-saving algorithm...")
        self.get_cost_saving_data()
        # st.dataframe(self.parameters['all_results'])

        create_shipment_window_vs_saving_plot(self.parameters['all_results'],self.parameters['best_params'])

        # Identify row with maximum cost savings
        max_savings_row = self.parameters['all_results'].loc[
            self.parameters['all_results']['Cost Savings'].idxmax()
        ].to_dict()
        chat_history.append({"Agent": f"Optimum results: {max_savings_row}"})

        # Agent for cost-saving data analysis
        agent = create_pandas_dataframe_agent(
            self.llm, self.parameters['all_results'],
            verbose=False, allow_dangerous_code=True,
            handle_parsing_errors=True, return_intermediate_steps=True
        )

        st.info("Analyzing the results...")
        # shipment_query = ("Analyze the data and provide quick insights and mention the best shipment window."
        #                   "Show plots for relevant insights."
        #                   "Use tool `python_repl_ast` to avoid `Could not parse LLM output:` error.")


        shipment_query = ("Share a quick insights by comparing Shipment Window against Total Shipments, Cost Savings and Total Shipment costs.",
                          "The insight should provide overview about how shipment window varies with these factors.",
                          "Avoid plots as plot is already there showing the trend, just provide a single or multi-line comment for each comparison.",
                          "Use `python_ast_repl_tool` to write a python script and  then print the results in order to pass it to final response.")

        final_answer = run_agent_query(agent, shipment_query, self.parameters['all_results'],max_attempts=3)
        chat_history.extend([{"Human": shipment_query}, {"Agent": final_answer}])

        # Determine shipment window
        user_window = None  # Replace with user input logic if needed
        self.parameters["window"] = int(user_window) if user_window else max_savings_row['Shipment Window']

        st.info(f"Consolidating orders for window {self.parameters['window']}...")
        self.consolidate_for_shipment_window()
        # st.dataframe(self.parameters['all_consolidated_shipments'])
        



        # Agent for consolidated shipment analysis
        # consolidated_agent = create_pandas_dataframe_agent(
        #     self.llm, self.parameters['all_consolidated_shipments'],
        #     verbose=False, allow_dangerous_code=True,
        #     handle_parsing_errors=True, return_intermediate_steps=True
        # )

        # shipment_freq_query = (
        #     "Analyze the data and provide quick insights."
        #     "Show plots for relevant insights."
        #     "Use tool `python_repl_ast` to avoid `Could not parse LLM output:` error."
        # )

        # shipment_freq_query = (
        #     "Analyze the data and provide quick insights. Include values in your response."
        # )

        # final_answer = run_agent_query(consolidated_agent, shipment_freq_query, self.parameters['all_consolidated_shipments'],max_attempts=3)
        # chat_history.extend([{"Human": shipment_freq_query}, {"Agent": final_answer}])

        # Compare pre- and post-consolidation results
        st.info("Comparing before and after consolidation...")
        comparison_df = self.compare_before_and_after_consolidation()
        comparison_results = comparison_df.to_dict()

        st.write("Before and After Consolidation Comparison:")
        # st.json(comparison_results)
        chat_history.append({"Agent": f"Comparison results: {comparison_results}"})
        create_calendar_heatmap_before_vs_after(self.parameters)




        return chat_history

    
