"""
Styling configurations for the Streamlit app 
    California Housing Price Predictor
"""

import streamlit as st

# Custom CSS for better styling
CUSTOM_CSS = """
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text_align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .prediction-box {
            background-color: #f0f8ff;
            padding: 2rem;
            border-radius: 10px;
            border: 2px solid #1f77b4;
            text-align: center;
            margin: 2rem 0;
        }
        .prediction-value {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
        }
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
"""


def apply_custom_styles():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def configure_page():
    """Configure Streamlit page settings
    """
    st.set_page_config(
        page_title='California Housing Price Predictor',
        page_icon='🏠',
        layout='wide',
        initial_sidebar_state='expanded'
    )
