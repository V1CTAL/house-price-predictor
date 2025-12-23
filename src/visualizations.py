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
    plot the feature importance chart.

    :param features: Description
    :type features: List[str]
    :param importances: Description
    :type importances: List[float]
    :param top_n: Description
    :type top_n: int
    :return: Description
    :rtype: Figure
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
    Docstring for plot_price_distribution

    :param data: Description
    :type data: pd.DataFrame
    :param predicted_price: Description
    :type predicted_price: Optional[float]
    :return: Description
    :rtype: Figure
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
    Plot housing prices geographically using an interactive map.
    Args:
        data (pd.DataFrame): DataFrame containing housing data with
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
    Plot the prediction with confidence intervals.
    Args:
        prediction_data (Dict[str, float]): Dictionary with keys:
            - 'prediction': float, the predicted price
            - 'lower_bound': float, lower bound of confidence interval
            - 'upper_bound': float, upper bound of confidence interval
    """
    # Create figure for the confidence interval plot
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
    Plot a comparison chart between user input features and dataset averages,
    Args:
        user_input (Dict[str, float]): User-provided feature values
        dataset_stats (pd.DataFrame): DataFrame containing dataset statistics
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
