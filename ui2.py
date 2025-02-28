import streamlit as st
import pandas as pd
import openai
import uuid
from PIL import Image
from MultiAgentGraph import multi_agent_graph
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages

def reset_app_state():
    """Reset the app state when the data source changes."""
    st.session_state.initialized = False
    st.session_state.pop('df', None)

def load_data_file(filename):
    """Load a CSV file with automatic date parsing."""
    try:
        df = pd.read_csv(filename)
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        return pd.read_csv(filename, parse_dates=date_columns, dayfirst=True)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

def check_openai_api_key(api_key):
    """Validate OpenAI API key."""
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True
    except openai.AuthenticationError:
        return False

def setup_sidebar():
    """Set up sidebar with API key input and data source selection."""
    st.sidebar.header("⚙️ Configuration")
    
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    
    data_files = {
        'Outbound_Data.csv': 'Data/Outbound_Data.csv',
        'Inventory_Batch.csv': 'Data/Inventory_Batch.csv',
        'Inbound_Data.csv': 'Data/Inbound_Data.csv'
    }
    
    st.sidebar.subheader("2. Data Source")
    data_source = st.sidebar.radio("Choose Data Source:", list(data_files.keys()), index=0)
    
    if st.session_state.get('current_data_source') != data_source:
        st.session_state.current_data_source = data_source
        reset_app_state()
    
    return api_key, data_files[data_source]

def display_sample_data():
    """Display sample data in an expander."""
    with st.expander("📊 View Sample Data"):
        df = st.session_state.df.copy()
        for col in df.select_dtypes(include=['datetime64']):
            df[col] = df[col].dt.strftime('%d-%m-%Y')
        st.dataframe(df.head(), use_container_width=True)

def main():
    st.set_page_config(
        page_title="Perrigo GenAI Answer Bot",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        logo = Image.open("Images/perrigo-logo.png")
        st.image(logo, width=120)
    except Exception:
        st.error("Logo image not found.")
    
    api_key, data_file = setup_sidebar()
    
    if not api_key:
        st.info("Please enter your OpenAI API key in the sidebar to get started.")
        st.stop()
    elif not check_openai_api_key(api_key):
        st.error("❌ Invalid API Key. Please check and try again.")
        st.stop()
    
    if 'df' not in st.session_state:
        st.session_state.df = load_data_file(data_file)
        if st.session_state.df is None:
            st.stop()
    
    display_sample_data()
    
    st.title("GenAI Answer Bot")
    st.subheader("💬 Ask Questions About Your Data")
    question = st.text_input("Please ask your question:", key="beginning_query")
    
    if st.button("Submit", key="langgraph_chat_submit"):
        if not question:
            st.error("Please enter a question before submitting.")
            return
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = {"messages": [HumanMessage(content=question)], "next": "supervisor"}
        
        counter = 0
        while state['next'] != 'FINISH' and counter < 10:
            current_state = multi_agent_graph.nodes[state['next']].invoke(state, config)
            
            st.markdown(f"""
                <div style="background-color: #eaecee; padding: 10px; border-radius: 10px; margin: 10px 0;">
                    <strong style="color: #2a52be;">{state['next'].upper()}:</strong>
                    <p style="color: #333;">{current_state['messages'][0].content}</p>
                </div>
            """, unsafe_allow_html=True)
            
            state['messages'] = add_messages(state['messages'], current_state['messages'])
            state['next'] = current_state['next']
            counter += 1
        
        st.success("Processing complete.")

if __name__ == '__main__':
    main()
