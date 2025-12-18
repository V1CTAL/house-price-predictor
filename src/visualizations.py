from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import seaborn as sns  # type: ignore
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore


def plot_feature_importance(
        features: List[str],
        importances: List[float],
        top_n: int = 10) -> mpl_figure.Figure:
    """
    Create a horizontal bar chart showing which features matter most for predictions.

    This visualization helps you understand what drives house prices in the model.
    For example, if 'median_income' has the highest bar, that means income is
    the strongest predictor of price in your model.

    Parameters:
        features: List of feature names sorted by importance (most important first)
                 Example: ['median_income', 'longitude', 'latitude', ...]
        importances: Corresponding importance scores (values between 0 and 1)
                    These scores show the relative contribution of each feature
        top_n: Number of top features to display (default 10)
              We limit this because showing all 20+ features can be overwhelming

    Returns:
        A matplotlib Figure object containing the bar chart, which can be
        displayed in Streamlit using st.pyplot(fig)

    Visual Design:
        - Horizontal bars make long feature names easier to read
        - Highest importance appears at the top (inverted y-axis)
        - Steel blue color provides good contrast without being harsh
    """
    # Create a figure with a reasonable size for web display
    # 10x6 inches provides good readability without takeing too much screen space
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract only the top N features and their scores
    # This keeps the visualization focused and uncluttered
    top_features: List[str] = features[:top_n]
    top_importances: List[float] = importances[:top_n]

    # Create horizontal bars (barh) rather than vertical (bar)
    # Horizontal orientation is better for reading feature names
    ax.barh(top_features, top_importances, color='steelblue')
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Most Important Features')

    # Invert y-axis so the most important feature appears at the top
    # THis matches how we naturally read ordered lists (most important first)
    ax.invert_yaxis()

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    return fig


def plot_price_distribution(
        data: pd.DataFrame,
        predicted_price: Optional[float] = None) -> mpl_figure.Figure:
    """
    Display a histogram showing how house prices are distributed across California.

    This visualization answers the question: "How common are different price ranges?"
    The height of each bar tells you how many houses fall into that price bracket.

    If you provide a predicted price, it will be marked with a red dashed line,
    allowing you to see where your prediction falls relative to all other houses.
    For example, if your prediction line is on the far right, you're predicting
    a price higher than most houses in the dataset.

    Parameters:
        data: DataFrame containing the housing data, must include 'median_house_value' column
        predicted_price: Optional price to mark on the distribution (e.g., from a prediction)
                        If None, just shows the overall distribution

    Returns:
        A matplotlib Figure showing the price distribution histogram

    Statistical Note:
        We use 50 bins to balance detail with clarity. Too few bins lose information,
        too many bins create a noisy, hard-to-read chart.
    """
    # Create a well-sized figure for the histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        data['median_house_value'],
        bins=50,
        edgecolor='black',
        alpha=0.7
    )

    # If a prediction was provided, mark it on the chart
    # This helps users understand "Is my prediction typical or unusual?"
    if predicted_price is not None:
        # axvline draws a vertical line at the specified x-value
        # The dashed red line makes it stand out from the histogram
        ax.axvline(
            predicted_price,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Your Prediction: ${predicted_price:,.0f}'
        )
        # Show the legend only when we have a prediction to label
        ax.legend()

    # Labe the axes clearly so users understand what they're looking at
    ax.set_xlabel('House Price ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of House Prices in California')

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    return fig


def plot_geographic_prices(
        data: pd.DataFrame,
        predicted_location: Optional[Dict[str, float]] = None) -> go.Figure:
    """
    Create an interactive map showing California housing data geographically.

    This is one of the most insightful visualizations because housing prices
    are heavily influenced by location. The map uses color to show population
    density, and you can hover over any point to see details about that area.

    The interactive nature means users can zoom into specific regions like
    San Francisco or Los Angeles to see fine-grained patterns.

    Parameters:
        data: DataFrame with geographic and housing data, must include:
             - 'latitude' and 'longitude' for positioning
             - 'population' for color coding
             - 'median_income' and 'housing_median_age' for hover information
        predicted_location: Optional dictionary with 'latitude' and 'longitude' keys
                           If provided, marks the user's property with a red star

    Returns:
        A Plotly Figure object with interactive map capabilities
        Users can zoom, pan, and hover for more details

    Design Choice:
        We use 'Viridis' color scale because it's colorblind-friendly and
        shows gradients clearly from low (purple) to high (yellow) values
    """
    # Create scatter plot on mapbox (interactive map background)
    # Each data point represents a geographic block group
    fig = px.scatter_mapbox(
        data,
        lat='latitude',
        lon='longitude',
        color='population',  # Color intensity represents population density
        # Extra info on hover
        hover_data=['median_income', 'housing_median_age'],
        color_continuous_scale='Viridis',  # Perceptually uniform color scheme
        size_max=15,  # Maximum marker size to prevent overcrowding
        zoom=5,  # Initial zoom level to show all of California
        title='California Housing Prices by Location'
    )

    # If the user made a prediction for a specific location, mark it distinctly
    # This helps them see their property in context of surrounding areas
    if predicted_location is not None:
        # Add a special marker trace for the user's property
        # go.Scattermapbox adds an additional layer on top of the main visualization
        fig.add_trace(go.Scattermapbox(
            lat=[predicted_location['latitude']],
            lon=[predicted_location['longitude']],
            mode='markers',
            marker=dict(size=20,
                        color='red',
                        symbol='star'
                        ),
            name='Your Property',
            text='Your Prediction'
        ))

    # Configure the map appearance
    # 'open-street-map' provides free, detailed base maps
    fig.update_layout(
        mapbox_style='open-street-map',
        height=600  # Adequate height for exploring map
    )

    return fig


def plot_prediction_confidence(
        prediction_data: Dict[str, float]) -> mpl_figure.Figure:
    """
    Visualize a prediction with its confidence interval as an error bar chart.

    This chart is crucial for understanding prediction uncertainty. The blue bar
    shows your predicted price, while the red error bars show the range where
    we're 95% confident the true price falls.

    Think of it like a weather forecast: "It will be 75°F, give or take 5 degrees."
    The prediction_data contains that central estimate and the "give or take" range.

    A narrow confidence interval means the model is very certain. A wide interval
    means there's more uncertainty, perhaps because the input features represent
    an unusual property the model hasn't seen much of during training.

    Parameters:
        prediction_data: Dictionary containing:
            - 'prediction': The central predicted price
            - 'lower_bound': Lower edge of 95% confidence interval
            - 'upper_bound': Upper edge of 95% confidence interval

    Returns:
        A matplotlib Figure showing the prediction with confidence bars

    Visual Elements:
        - Blue horizontal bar: The point estimate (our best guess)
        - Red error bars with caps: The uncertainty range
        - The caps on the error bars make the bounds clearly visible
    """
    # Create a compact figure since we're only showing one prediction
    fig, ax = plt.subplots(figsize=(10, 4))

    # Extract the prediction components for clearer code
    prediction: float = prediction_data['prediction']
    lower: float = prediction_data['lower_bound']
    upper: float = prediction_data['upper_bound']

    # Draw the main prediction as a horizontal bar
    # Using barh with a single category creates a clean, simple visualization
    ax.barh(['Price'],
            [prediction],
            color='steelblue',
            label='Predicted Price'
            )

    # Add error bars to show the confidence interval
    # The xerr parameter expects the distance from the center to each bound
    # We provide this as [[left_distance], [right_distance]]
    ax.errorbar(
        [prediction],  # x-position (center of error bar)
        ['Price'],  # y-position (same category)
        xerr=[[prediction - lower], [upper - prediction]],  # Asymmetric errors
        fmt='none',  # Don't draw markers at the center
        color='red',  # Red for visibility against blue bar
        capsize=10,  # Size of end caps
        capthick=2,  # Thickness of the end caps
        label='95% Confidence Interval'
    )

    # Label everything clearly
    ax.set_xlabel('Price ($)')
    ax.set_title('Price Prediction with Confidence Interval')
    ax.legend()

    # Use plain number format (no scientific notation) for prices
    # ticklabel_format prevents "$2.5e5" and shows "$250,000" instead
    ax.ticklabel_format(style='plain', axis='x')

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    return fig


def plot_comparison_chart(
        user_input: Dict[str, float],
        dataset_stats: pd.DataFrame) -> mpl_figure.Figure:
    """
    Create a side-by-side comparison of user inputs versus dataset averages.

    This visualization helps users understand: "Is my property typical or unusual?"
    For example, if your median_income bar is much higher than the average bar,
    you're looking at a wealthier area than typical.

    This context is valuable because it helps explain why a prediction might be
    high or low. If most of your input values exceed the averages, a high
    predicted price makes sense.

    Parameters:
        user_input: Dictionary of feature names to values that the user provided
                   Example: {'median_income': 5.0, 'housing_median_age': 15, ...}
        dataset_stats: DataFrame containing the full dataset for calculating averages
                      Must include columns matching the keys in user_input

    Returns:
        A matplotlib Figure with grouped bar chart comparing input vs average

    Limitation:
        This only works for numeric features. Categorical features like
        ocean_proximity are excluded from this comparison.
    """
    # Extract feature names and user-provided values
    features: List[str] = list(user_input.keys())
    user_values: List[float] = list(user_input.values())

    # Calculate average values from the dataset for matching features
    # We use a list comprehension with a condition to handle missing columns gracefully
    avg_values: List[float] = [
        dataset_stats[f].mean()
        for f in features
        if f in dataset_stats.columns
    ]

    # Create figure for the comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up x-axis positions for the grouped bars
    x = range(len(features))
    width: float = 0.35  # Width of each bar (0.35 leaves nice spacing)

    # Draw two sets of bars side by side
    # The offset of width/2 positions them next to each other rather than overlapping
    ax.bar(
        [i - width / 2 for i in x],
        user_values,
        width,
        label='Your Input',
        color='steelblue'
    )
    ax.bar([i - width / 2 for i in x],
           avg_values,
           width,
           label='Dataset Average',
           color='orange'
           )

    # Configure axis labels and styling
    ax.set_xlabel('Features')
    ax.set_ylabel('Values')
    ax.set_title('Your Property vs California Average')
    ax.set_xticks(x)

    # Rotate feature names fr readability
    # ha='right' aligns the text properly after rotation
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    return fig
