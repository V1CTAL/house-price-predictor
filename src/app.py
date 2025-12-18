import streamlit as st
import pandas as pd
import numpy as np
from predictor import HousingPredictor
from visualizations import *


# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon='🏠',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4
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
        color: #1f77b4
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data


@st.cache_resource
def load_model():
    return HousingPredictor('./models/housing_price_model.pkl')


@st.cache_data
def load_data():
    return pd.read_csv('./data/housing_cleaned.csv')


model = load_model()
data = load_data()

# Sidebar navigation
st.sidebar.title('🏠 Navigation')
page = st.sidebar.radio(
    'Go to',
    ['Home', 'Price Predictor', 'Data Explorer', 'Model Insights', 'About']
)

# ==================== HOME PAGE ====================
if page == 'Home':
    st.markdown('<h1 class="main-header">🏠 California Housing Price Predictor</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Housing Price Predictor App!
    This machine learning application predicts house prices in California based
    on various features like location, median income, house age, and more.

    #### 🎯 What You Can Do:
    - **Predict Prices**: Enter property details and get instant price predictions
    - **Explore Data**: Visualize California housing market trends
    - **Mode Insights**: Understand what factors matter most for house prices

    #### 📊 About the Model:
    - **Algorithm**: Random Forest Regressor
    - **Training Data**: 20,000+ California housing records
    - **Accuracy**: R² Score of ~0.81 (explains 81 of price variance)
    - **Features**: Location, income, house age, rooms, and more

    #### 🚀 Get Started:
    Use the sidebar to navigate through different sections!
    """)

    # Show some quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric('Total Properties', f'{len(data):,}')

    with col2:
        avg_price = data['median_house_value'].mean()
        st.metric('Average Price', f'${avg_price:,.0f}')

    with col3:
        max_price = data['median_house_value'].max()
        st.metric('Highest Price', f'${max_price:,.0f}')

    with col4:
        min_price = data['median_house_value'].min()
        st.metric('Lower Price', f'${min_price:,.0f}')

    # Quick visualization
    st.subheader('📍 Housing Prices Across California')
    fig = plot_price_distribution(data)
    st.pyplot(fig)

# ==================== PRICE PREDICTOR PAGE ====================
elif page == 'Price Predictor':
    st.markdown('<h1 class="main-header">💰 Predict House Price</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    Enter the property details below to get an estimated price. All fields are required.
    """)

    # Create input form
    with st.form('prediction_form'):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('📍 Location')
            longitude = st.number_input(
                'Longitude',
                min_value=-124.0,
                max_value=-114.0,
                value=-118.0,
                step=0.01,
                help="Longitude coordinate (-124 to -114)"
            )

            latitude = st.number_input(
                'Latitude',
                min_value=32.0,
                max_value=42.0,
                value=34.0,
                step=0.01,
                help='Latitude coordinates (32 to 42)'
            )

            st.subheader('🏡 Property Details')
            housing_median_age = st.slider(
                'House Age (years)',
                min_value=1,
                max_value=52,
                value=25,
                help='Median age of houses in the block'
            )

            total_rooms = st.number_input(
                'Total Rooms',
                min_value=1,
                max_value=40_000,
                value=2000,
                step=100,
                help='Total number of rooms in the block'
            )

        with col2:
            st.subheader('👥 Demographics')
            population = st.number_input(
                'Population',
                min_value=1,
                max_value=40_000,
                value=1500,
                step=100,
                help='Total population in the block'
            )

            households = st.number_input(
                'Households',
                min_value=1,
                max_value=7000,
                value=500,
                step=50,
                help='Number of households in the block'
            )

            median_income = st.number_input(
                'Median Income (in $10,000s)',
                min_value=0.5,
                max_value=15.0,
                value=3.5,
                step=0.1,
                help='Median income in units of $10,000 (e.g., 3.5 = $35,000)'
            )

            st.subheader('🌊 Ocean Proximity')
            ocean_proximity = st.selectbox(
                'Distance to Ocean',
                ['NEAR BAY', 'INLAND', '1<H OCEAN', 'NEAR OCEAN', 'ISLAND']
            )

        # Submit button
        submitted = st.form_submit_button(
            '🔮 Predict Price', use_container_width=True)

        if submitted:
            # Calculated engineered features
            bedrooms_per_room = 0.2  # Approximate
            rooms_per_household = total_rooms / households
            population_per_household = population / households

            # Prepare features (must match training data)
            features = {
                'longitude': longitude,
                'latitude': latitude,
                'housing_median_age': housing_median_age,
                'total_rooms': total_rooms,
                'population': population,
                'households': households,
                'median_income': median_income,
                'bedrooms_per_room': bedrooms_per_room,
                'rooms_per_household': rooms_per_household,
                'population_per_household': population_per_household
            }

            # Add one-hot encoded ocean proximity
            for prox in ['NEAR BAY', 'INLAND', '1<H OCEAN', 'NEAR OCEAN', 'ISLAND']:
                features[f'ocean_{prox.replace(" ", "_")}'] = 1 if ocean_proximity == prox else 0

            # Make prediction
            with st.spinner('Calculating price...'):
                prediction_result = model.predict_with_confidence(features)

            # Display result
            st.markdown('<div class="prediction-bo">', unsafe_allow_html=True)
            st.markdown('### 🎯 Predicted House Price')
            st.markdown(f'<p class="prediction-value">${prediction_result["prediction"]:,.0f}</p>',
                        unsafe_allow_html=True)

            st.markdown(f"""
            **95% Confidence Interval:**
            ${prediction_result['lower_bound']:,.0f} - ${prediction_result['upper_bound']:,.0f}
            """)
            st.markdown('</div>', unsafe_allow_html=True)

            # Visualize confidence interval
            fig = plot_prediction_confidence(prediction_result)
            st.pyplot(fig)

            # Show where this fall in the distribution
            st.subheader('📊 How does this compare?')
            fig2 = plot_price_distribution(
                data, prediction_result["prediction"])
            st.pyplot(fig2)

# ==================== DATA EXPLORER PAGE ====================
elif page == "Data Explorer":
    st.markdown('<h1 class="main-header">📊 Explore the Data</h1>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["📈 Statistics", " Geographic View", "🔍 Raw Data"])

    with tab1:
        st.subheader('Dataset Statistics')
        st.dataframe(data.describe())

        st.subheader('Price Distribution')
        fig = plot_price_distribution(data)
        st.pyplot(fig)

        st.subheader('Feature Correlations')
        fig, ax = plt.subplots(figsize=(12, 8))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader('Interactive Map of California Housing')
        st.info('💡 Zoom in/out and hover over points to see details!')

        # Sample data for performance (plotting 20k points can be slow)
        sample_data = data.sample(min(5000, len(data)))
        fig = plot_geographic_prices(sample_data)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader('Raw Dataset')
        st.dataframe(data, use_container_width=True)

        st.download_button(
            label='📥 Download Data as CSV',
            data=data.to_csv(index=False).encode('utf-8'),
            file_name='california_housing_data.csv',
            mime='text/csv'
        )

# ==================== MODEL INSIGHTS PAGE ====================
elif page == 'Model Insights':
    st.markdown('<h1 class="main-header"🔬 Model Insights</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    Understanding what drives house prices in California.
    """)

    # Feature importance
    st.subheader('🎯 Most Important Features')
    importance_data = model.get_feature_importance()

    if importance_data:
        fig = plot_feature_importance(
            importance_data['features'],
            importance_data['importances']
        )
        st.pyplot(fig)

        st.markdown("""
        **Key Insights:**
        - **Median Income**: The strongest predictor of house prices
        - **Location**: Longitude and latitude significantly impact price
        - **Ocean Proximity**: Properties near the coast are more valuable
        - **Property Age**: Newer homes tend to be more expensive
        """)

    # Model performance
    st.subheader('📊 Model Performance')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('R² Score', '0.81', help='Model explains 81%'
                  'of price variance')

    with col2:
        st.metric('Mean Absolute Error',  '$49,000',
                  help='Average prediction error')

    with col3:
        st.metric('Accuracy', 'Very Good',
                  help='Model perform well on test data')

# ==================== ABOUT PAGE ====================
elif page == 'About':
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>',
                unsafe_allow_html=True)

st.markdown("""
### 🎓 Project Overview

This is a **Machine Learning** portfolio project that predicts California housing prices
using real-world data and industry-standard techniques.

### 🛠️ Tech Stack
- **Python 3.13.11**
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning modeling
- **Matplotlib & Seaborn**: Data visualization
- **Streamlit**: Interactive web application
- **Plotly**: Interactive geographic maps

### 📚 Dataset
The California Housing Dataset contains 20,640 records from the 1990 California
including:
- Geographic location (latitude/longitude)
- Housing characteristics (age, rooms, bedrooms)
- Population demographics
- Economic indicators (median income)
- Proximity to ocean

### 🤖 Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Why Random Forest?**
  - Handles non-linear relationships well
  - Robust to outliers
  - Provides feature importance
  - Less prone to overfitting than single decision trees

### 📈 Model Performance
- Trained on 80% of data (16,512 properties)
- Tested on 20% of data (4,128 properties)
- R² Score: 0.81 (very good predictive power)
- Mean Absolute Error: ~R$49,000

### 👨‍💻 Developer
**Your Name**
Aspiring Data Scientist | Python Developer

[GitHub](your-github-url) | [LinkedIn](your-linkedin-url)

---

### 🚀 Want to Learn More?
Check out the project repository on GitHub for:
- Complete source code
- Jupyter notebooks with EDA
- Model training process
- Setup instructions
""")
