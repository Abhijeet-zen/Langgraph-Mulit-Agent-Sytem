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



# def display_saved_plot(plot_path: str) -> None:
#     """
#     Loads and displays a saved plot from the given path in a Streamlit app.

#     Args:
#         plot_path (str): Path to the saved plot image.
#     """
#     if os.path.exists(plot_path):
#         st.image(plot_path, caption="",)
#     else:
#         st.error(f"Plot not found at {plot_path}")



def display_saved_plot(plot_path: str,):

    """
    Loads and displays a saved plot from the given path in a Streamlit app with a highlighted background.

    Args:
        plot_path (str): Path to the saved plot image.
        bg_color (str): Background color for the image container.
        padding (str): Padding inside the image container.
        border_radius (str): Border radius for rounded corners.
    """

    bg_color: str = "#f0f2f6"
    padding: str = "5px"
    border_radius: str = "10px"
    if os.path.exists(plot_path):
        # Apply styling using markdown with HTML and CSS
        st.markdown(
            f"""
            <style>
                .image-container {{
                    background-color: {bg_color};
                    padding: {padding};
                    border-radius: {border_radius};
                    display: flex;
                    justify-content: center;
                }}
            </style>
            <div class="image-container">
                <img src="data:image/png;base64,{get_base64_image(plot_path)}" style="max-width:100%; height:auto;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error(f"Plot not found at {plot_path}")

def get_base64_image(image_path: str) -> str:
    """
    Converts an image to a base64 string.

    Args:
        image_path (str): Path to the image.

    Returns:
        str: Base64-encoded image.
    """
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


