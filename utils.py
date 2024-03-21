import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from vars import COL_MAP


def standardize_column_names(df):
    """
    Standardize column names in the DataFrame based on a predefined mapping.

    Parameters:
    - df: The DataFrame whose columns need to be standardized.
    - column_mapping: A dictionary mapping from potential column names to a standardized name.

    Returns:
    - The DataFrame with standardized column names.
    """
    df.columns = df.columns.map(lambda x: COL_MAP.get(x, x))
    return df


def calculate_truck_fill(vims_tonnage, truck_factor_tonnage):
    return (vims_tonnage / truck_factor_tonnage) * 100


def preprocessdata(df):
    """
    Preprocess the data.

    Parameters:
    - df: The DataFrame that needs to be processed.

    Returns:
    - The DataFrame with formatted data.
    """
    if 'time_full' in df.columns:
        df['time_full'] = pd.to_datetime(df['time_full'])
    df['Month'] = df['time_full'].dt.month_name()
    df['Month Order'] = df['time_full'].dt.month

    # Drop 'Truck fill' column if it exists
    if 'Truck fill' in df.columns:
        df = df.drop(columns=['Truck fill'])

    # Calculate 'Truck fill' based on 'VIMS_tonnage' and 'truck_factor_tonnage'
    df['Truck fill'] = calculate_truck_fill(df['VIMS_tonnage'], df['truck_factor_tonnage'])

    return df


def filter_data(df, material, dump, shift):
    # Filter by material and dump
    df = df[df['Material'].isin(material) & df['Dump'].isin(dump)]

    # Define time ranges for shifts
    shift_ranges = {
        '7am-7pm': ('07:00', '19:00'),
        '7pm-7am': ('19:00', '07:00')
    }

    # Filter by shift
    if '7am-7pm' in shift:
        df = df[(df['time_full'].dt.time >= pd.to_datetime(shift_ranges['7am-7pm'][0]).time()) &
                (df['time_full'].dt.time < pd.to_datetime(shift_ranges['7am-7pm'][1]).time())]
    if '7pm-7am' in shift:
        df = df[(df['time_full'].dt.time >= pd.to_datetime(shift_ranges['7pm-7am'][0]).time()) |
                (df['time_full'].dt.time < pd.to_datetime(shift_ranges['7pm-7am'][1]).time())]

    return df


def generate_plot(df, shovel_of_interest, target_std_dev=5, mean_fill=100):
    actual_data = df['Truck fill']
    (x_range, actual_distribution_y, desired_distribution_y, actual_productivity,
     desired_productivity, productivity_difference) = get_dist_data(actual_data, mean_fill, target_std_dev)
    month_of_interest = f'{df["Month"].iloc[0]}-{df["Month"].iloc[-1]}'

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_range,
                             y=actual_distribution_y,
                             mode='lines',
                             name=f'Actual Distribution for {shovel_of_interest} - {month_of_interest}',
                             line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_range,
                             y=desired_distribution_y,
                             mode='lines',
                             name='Desired Distribution',
                             line=dict(color='#00B7F1')))

    fig.add_trace(go.Scatter(x=[np.mean(actual_data),np.mean(actual_data)],
                             y=[0, max(actual_distribution_y)],
                             mode='lines',
                             name='Actual Mean',
                             line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=[mean_fill, mean_fill],
                             y=[0, max(desired_distribution_y)],
                             mode='lines',
                             name='Desired Mean',
                             line=dict(color='#00B7F1', dash='dash')))

    mean_std_text = (f"<b>Actual Mean:</b> {np.mean(actual_data):.2f}%<br>"
                     f"<b>Actual Std Dev:</b> {np.std(actual_data):.2f}%<br>"
                     f"<b>Desired Mean:</b> {mean_fill}%<br>"
                     f"<b>Desired Std Dev:</b> {target_std_dev}%<br>"
                     f"<b>Shovel of Interest:</b> {shovel_of_interest}<br>"
                     f"<b>Productivity Difference:</b> {productivity_difference:.2f}%")

    fig.add_annotation(
        text=mean_std_text,
        align='left',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=0.02,
        y=0.98,
        bordercolor='black',
        borderwidth=1,
        bgcolor='white',
        xanchor='left',
        yanchor='top'
    )

    fig.update_layout(
        title=f'Actual vs Desired Truck Fill Distribution ({shovel_of_interest} - {month_of_interest})',
        xaxis_title='Truck Fill %',
        yaxis_title='Probability Density',
        xaxis=dict(range=[50, 150], dtick=10),
        yaxis=dict(range=[0, max(max(actual_distribution_y), max(desired_distribution_y)) * 1.2]),
        legend_title='Legend',
        legend=dict(x=1, y=1, bgcolor='rgba(255,255,255,0.5)'),
    )

    return fig


def get_dist_data(actual_data, mean_fill, target_std_dev):
    x_range = np.linspace(min(actual_data), max(actual_data), 200)
    actual_distribution_y = norm.pdf(x_range, np.mean(actual_data), np.std(actual_data))
    desired_distribution_y = norm.pdf(x_range, mean_fill, target_std_dev)

    actual_productivity = np.mean(actual_data)
    desired_productivity = mean_fill
    productivity_difference = max(0, desired_productivity - actual_productivity)

    return x_range, actual_distribution_y, desired_distribution_y, actual_productivity, desired_productivity, productivity_difference