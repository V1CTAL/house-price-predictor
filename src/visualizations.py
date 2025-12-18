import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore


def plot_feature_importance(features, importances, top_n=10):
    """
    Create a bar chart of feature importance
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get top N features
    top_features = features[:top_n]
    top_importances = importances[:top_n]

    ax.barh(top_features, top_importances, color='steelblue')
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.invert_yaxis()  # Highest importance on top

    plt.tight_layout()
    plt.show()

    return fig


def plot_price_distribution(data, predicted_price=None):
    """
    Plot distribution of house prices with optional prediction marker
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(data['median_house_value'], bins=50, edgecolor='black', alpha=0.7)

    if predicted_price:
        ax.axvline(predicted_price, color='red', linestyle='--', linewidth=2,
                   label=f'Your Prediction: ${predicted_price:,.0f}')
        ax.legend()

    ax.set_xlabel('House Price ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of House Prices in California')

    plt.tight_layout()
    plt.show()

    return fig


def plot_geographic_prices(data, predicted_location=None):
    """
    Create interactive map of California housing prices
    """
    fig = px.scatter_mapbox(
        data,
        lat='latitude',
        lon='longitude',
        color='population',
        hover_data=['median_income', 'housing_median_age'],
        color_continuous_scale='Viridis',
        size_max=15,
        zoom=5,
        title='California Housing Prices by Location'
    )

    # If user provided location, mark it
    if predicted_location:
        fig.add_trace(go.Scattermapbox(
            lat=[predicted_location['latitude']],
            lon=[predicted_location['longitude']],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='Your Property',
            text='Your Prediction'
        ))

    fig.update_layout(
        mapbox_style='open-street-map',
        height=600
    )

    return fig


def plot_prediction_confidence(prediction_data):
    """
    Visualize prediction with confidence interval
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    prediction = prediction_data['prediction']
    lower = prediction_data['lower_bound']
    upper = prediction_data['upper_bound']

    ax.barh(['Price'], [prediction],
            color='steelblue', label='Predicted Price')
    ax.errorbar([prediction], ['Price'],
                xerr=[[prediction - lower], [upper - prediction]],
                fmt='none', color='red', capsize=10, capthick=2,
                label='95% Confidence Interval')

    ax.set_xlabel('Price ($)')
    ax.set_title('Price Prediction with Confidence Interval')
    ax.legend()
    ax.ticklabel_format(style='plain', axis='x')

    plt.tight_layout()
    plt.show()

    return fig


def plot_comparison_chart(user_input, dataset_stats):
    """
    Compare use's inputs with dataset averages
    """
    features = list(user_input.keys())
    user_values = list(user_input.values())
    avg_values = [dataset_stats[f].mean()
                  for f in features if f in dataset_stats.columns]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(features))
    width = 0.35

    ax.bar([i - width / 2 for i in x], user_values,
           width, label='Your Input', color='steelblue')
    ax.bar([i - width / 2 for i in x], avg_values, width,
           label='Dataset Average', color='orange')

    ax.set_xlabel('Features')
    ax.set_ylabel('Values')
    ax.set_title('Your Property vs California Average')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()

    return fig
