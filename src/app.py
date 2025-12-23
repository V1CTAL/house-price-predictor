"""
California Housing Price Predictor - Main Application
This Streamlit application allows users to explore housing data, predict prices using a trained model,
view prediction history, and gain insights into the model's performance.
"""

import os
import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Local imports
from database import HousingPriceDB
from predictor import HousingPredictor
from styles import configure_page, apply_custom_styles
from pages import (
    render_home_page,
    render_price_predictor_page,
    render_data_explorer_page,
    render_prediction_history_page,
    render_model_insights_page,
    render_about_page
)

# Load environment variables
load_dotenv()

# Configure page (must be first Streamlit command)
configure_page()

# Apply custom CSS
apply_custom_styles()

# Define paths
MODEL_PATH = Path('./models/housing_price_model.pkl')
DATA_PATH = Path('./data/housing_cleaned.csv')


@st.cache_resource
def load_model():
    """Load the trained housing price prediction model from disk"""
    return HousingPredictor(MODEL_PATH)


@st.cache_resource
def get_database():
    """Initialize database connection"""
    try:
        db = HousingPriceDB(
            dbname=os.getenv('DB_NAME', 'housing_db'),
            user=os.getenv('DB_USER', 'housing_user'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432))
        )
        return db

    except Exception as e:
        st.error(f'Database connection failed: {e}')
        return None


@st.cache_resource
def load_data():
    """Load the cleaned housing dataset from CSV file"""
    return pd.read_csv(DATA_PATH)


# Load Resources
model = load_model()
db = get_database()
data = load_data()

# Map page names to functions and their required arguments
PAGES = {
    'Home': lambda: render_home_page(data),
    'Price Predictor': lambda: render_price_predictor_page(model, db, data),
    'Data Explorer': lambda: render_data_explorer_page(data),
    'Model Insights': lambda: render_model_insights_page(model),
    'Prediction History': lambda: render_prediction_history_page(db),
    'About': render_about_page,
}

# Sidebar Setup
st.sidebar.title('🏠 Navigation')
selection = st.sidebar.radio('Go to', list(PAGES.keys()))

# Database Status (Ternary Operator)
status_msg = "✅ Database Connected" if db else "⚠️ Database Offline"
render_status = st.sidebar.success if db else st.sidebar.warning
render_status(status_msg)

# Execute the selected page
PAGES[selection]()
