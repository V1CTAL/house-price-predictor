"""
Page components and layouts for the Streamlit app
    California Housing Price Predictor
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from typing import Dict, Any, Optional
from visualizations import *


def render_home_page(data: pd.DataFrame):
    """Render the home page"""
    st.markdown('<h1 class="main-header">🏠 California Housing Price Predictor</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Housing Price Predictor App!
    This machine learning application predicts house prices in California based
    on various features like location, median income, house age and more.
                
    #### 🎯 What You Can Do:
    - **Predict Prices**: Enter property details and get instant price predictions
    - **Explore Data**: Visualize California housing market trends
    - **Model Insights**: Understand what factors matter most for house prices
    - **Prediction History**: View all saved predictions from the database
    
    #### 📊 About the Model:
    **Algorithm**: Random Forest Regressor
    **Training Data**: 20, 000+ California housing records
    **Accuracy**: R² Score of ~0.81 (explains 81% of price variance)
    **Features**: Location, income, house age, rooms, and more

    #### 🚀 Get Started
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
        st.metric('Lowest Price', f'${min_price:,.0f}')

    # Quick visualization
    st.subheader('🔍 Housing Prices Across California')
    fig = plot_price_distribution(data)
    st.pyplot(fig)


def render_price_predictor_page(model, db, data):
    """Render the price predictor page"""
    st.markdown('<h1 class="main-header">💰 Predict House Price</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    Enter the property details below to get and estimated price. All fields are required.
    """)

    # Create input form
    with st.form('prediction_form'):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('📍 Location')
            address = st.text_input(
                'Property Address (Optional)',
                placeholder='123 Main St, San Francisco, CA',
                help='Enter the property address for record keeping'
            )

            longitude = st.number_input(
                'Longitude',
                min_value=-124.0,
                max_value=-114.0,
                value=-118.0,
                step=0.01,
                help='Longitude coordinate (-124 to -114)'
            )

            latitude = st.number_input(
                'Latitude',
                min_value=32.0,
                max_value=42.0,
                value=34.0,
                step=0.05,
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
            with col2:
                households = st.number_input(
                    'Households',
                    min_value=1,
                    max_value=7000,
                    value=500,
                    step=50,
                    help='Number of households in the block'
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
                median_income = st.number_input(
                    'Median Income (in $10,000s)',
                    min_value=0.5,
                    max_value=15.0,
                    value=3.5,
                    step=0.1,
                    help='Median income in units of $10,000 (e.g., 3.5 = $35.000)'
                )
            with col1:
                st.subheader('🌊 Ocean Proximity')
                ocean_proximity = st.selectbox(
                    'Distance to Ocean',
                    ['NEAR BAY', 'INLAND', '1<H OCEAN', 'NEAR OCEAN', 'ISLAND']
                )

            # Checkbox to save to database
            save_to_db = st.checkbox(
                '💾 Save prediction to database',
                value=True,
                help='Save this property and prediction for future reference'
            )

            # Submit button
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submitted = st.form_submit_button(
                    '🔮 Predict Price', use_container_width=True
                )

            if submitted:
                # Calculated engineered features
                bedrooms_per_room = 0.2
                rooms_per_household = total_rooms / households
                population_per_household = population / households

                # Prepare features
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

                # Map Ocean Proximity
                ocean_mapping = {
                    '1<H OCEAN': 0,
                    'INLAND': 1,
                    'ISLAND': 2,
                    'NEAR BAY': 3,
                    'NEAR OCEAN': 4
                }
                features['ocean_proximity'] = ocean_mapping[ocean_proximity]

                # Make prediction
                with st.spinner('Calculating price...'):
                    prediction_result = model.predict_with_confidence(
                        features)

                # Save to database if requested
                property_id = None
                prediction_id = None

                if save_to_db and db:
                    try:
                        property_data = {
                            'address': address if address else f'Lat: {latitude}, Lon: {longitude}',
                            'bedrooms': int(total_rooms * bedrooms_per_room),
                            'bathrooms': 2.0,
                            'square_feet': int(total_rooms * 400),
                            'lot_size': int(total_rooms * 500),
                            'year_built': 2024 - housing_median_age,
                            'zip_code': '00000',
                            'latitude': latitude,
                            'longitude': longitude,
                            'actual_price': None
                        }

                        property_id = db.insert_property(property_data)
                        prediction_id = db.save_prediction(
                            property_id=property_id,
                            predicted_price=prediction_result['prediction'],
                            model_version='v1.0',
                            confidence=0.95
                        )

                        st.success(
                            f'✅ Saved to database! Property ID: {property_id}')

                    except Exception as e:
                        st.error(f'Failed to save to database: {e}')

                st.markdown('<div class"prediction-box">',
                            unsafe_allow_html=True)
                st.markdown('### 🎯 Predicted House Price')
                st.markdown(f'<p class="prediction-value">${prediction_result['prediction']:,.0f}</p>',
                            unsafe_allow_html=True)

                st.markdown(f"""
                **95% Confidence Interval:**
                ${prediction_result['lower_bound']:,.0f} - ${prediction_result['upper_bound']:,.0f}
                """)

                if property_id:
                    st.markdown(
                        f'**Property ID:** {property_id} | **Prediction ID:** {prediction_id}')

                st.markdown('</div>', unsafe_allow_html=True)

                # Visualize confidence interval
                fig = plot_prediction_confidence(prediction_result)
                st.pyplot(fig)

                # Show where this fall in the distribution
                st.subheader('📊 How does this compare?')
                fig2 = plot_price_distribution(
                    data, prediction_result['prediction'])
                st.pyplot(fig2)


def render_data_explorer_page(data: pd.DataFrame):
    """Render the data explorer page"""
    st.markdown('<h1 class="main-header">📊 Explore the Data</h1>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ['📈 Statistics', '🗺️ Geographic view', '🔍 Raw Data'])

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

        sample_size = min(5000, len(data))
        sample_data = data.sample(sample_size)
        fig = plot_geographic_prices(sample_data)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader('Raw Dataset')
        st.dataframe(data, use_container_width=True)

        csv_data = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥  Download Data as CSV',
            data=csv_data,
            file_name='california_housing_data.csv',
            mime='text/csv'
        )


def render_prediction_history_page(db):
    """Render the prediction history page"""
    st.markdown('<h1 class="main-header">📜 Prediction History</h1>',
                unsafe_allow_html=True)

    if not db:
        st.error(
            'Database connection not available. Please check your configuration.')
    else:
        try:
            history_df = db.get_all_predictions_history(limit=100)

            if len(history_df) == 0:
                st.info('No predictions saved yet. Make some predictions first!')
            else:
                st.success(f'Found {len(history_df)} properties in database')

                st.subheader('Recent Properties')
                st.dataframe(history_df, use_container_width=True)

                st.subheader('📊 Model Performance')
                stats = db.get_model_performance_stats('v1.0')

                if stats:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric('Mean Absolute Error',
                                  f'${stats['mae']:,.0f}')
                    with col2:
                        st.metric('MAPE', f'{stats['mape']:.2f}$')

                    with col3:
                        st.metric('RMSE', f'${stats['rmse']:,.0f}')

                    with col4:
                        st.metric('Total Predictions',
                                  stats['total_predictions'])
                else:
                    st.info('No performance metrics available yet.')

        except Exception as e:
            st.error(f'Error loading prediction history: {e}')


def render_model_insights_page(model):
    """Render the model insights page"""
    st.markdown('<h1 class="main-header">🔬 Model Insights</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    Understanding what drives house prices in California.
    """)

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

    st.subheader('📊 Model Performance')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('R2 Score', '0.81',
                  help='Model explains 81% of price variance')

    with col2:
        st.metric('Mean Absolute Error', '$49,000',
                  help='Average prediction error')

    with col3:
        st.metric('Accuracy', 'Very Good',
                  help='Model performs well on test data')


def render_about_page():
    """Render the about page"""
    st.markdown('<h1 class"main-header">ℹ️ About This Project</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    ### 🎓 Project Overview

    This is a **Machine Learning** portfolio project that predicts California 
    housing prices using real-world data and industry-standard techniques.
                
    ### 🛠️ Tech Stack
    - **Python 3.13**
    - **Pandas & NumPy**: Data manipulation and analysis
    - **Scikit-learn**: Machine learning modeling
    - **Matplotlib & Seaborn**: Data visualization
    - **Streamlit**: Interactive web application
    - **Plotly**: Interactive geographic maps
    - **PostgreSQL**: Database for storing predictions
    - **psycopg2**: PostgreSQL adapter
    - **python-dotenv**: Environment variable management
                
    ### 📚 Dataset
    The California Housing Dataset contains 20,640 records from the 1990 California census,
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
    - Mean Absolute Error: ~$49,000

    ### 👨‍💻 Developer
    **Victal**
    Aspiring Data Scientist | Python Developer

    [GitHub](https://github.com/V1CTAL) | [LinkedIn](your-linkedin-url)

    ---

    ### 🚀 Want to Learn More?
    Check out the project repository on GitHub for:
    - Complete source code
    - Jupyter notebooks with EDA
    - Model training process
    - Setup instructions
    """)
