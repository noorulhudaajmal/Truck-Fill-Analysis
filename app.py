import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import re

st.set_page_config(layout="wide")
st.title("Truck Fill Analysis")


def standardize_column_names(df, column_mapping):
    """
    Standardize column names in the DataFrame based on a predefined mapping.
    
    Parameters:
    - df: The DataFrame whose columns need to be standardized.
    - column_mapping: A dictionary mapping from potential column names to a standardized name.
    
    Returns:
    - The DataFrame with standardized column names.
    """
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    return df

def calculate_truck_fill(vims_tonnage, truck_factor_tonnage):
    return (vims_tonnage / truck_factor_tonnage) * 100


def extract_month_name(in_string):
    """
    This function uses a regular expression pattern that 
    matches any of the month names followed by an optional 
    space and a four-digit year.
    
    Parameters:
    - in_string: A string containing a month name followed by 
                 an optional space and a four-digit year.
    
    Returns:
    The month name.
    """
    # Regular expression pattern to match month names followed by optional space and year
    pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ ]?\d{4}'
    
    match = re.search(pattern, in_string)
    if match:
        # Extract the month name
        return match.group(1)
    else:
        return in_string
    
    
def generate_plot(df, shovel_of_interest, target_std_dev=5, mean_fill=100):
    # Drop 'Truck fill' column if it exists
    if 'Truck fill' in df.columns:
        df = df.drop(columns=['Truck fill'])
    
    # Calculate 'Truck fill' based on 'VIMS_tonnage' and 'truck_factor_tonnage'
    df['Truck fill'] = calculate_truck_fill(df['VIMS_tonnage'], df['truck_factor_tonnage'])
    
    actual_data = df['Truck fill']
    month_of_interest = f'{df["Month"].iloc[0]}-{df["Month"].iloc[-1]}'
    
    x_range = np.linspace(min(actual_data), max(actual_data), 200)
    actual_distribution_y = norm.pdf(x_range, np.mean(actual_data), np.std(actual_data))
    desired_distribution_y = norm.pdf(x_range, mean_fill, target_std_dev)
    
    actual_productivity = np.mean(actual_data)
    desired_productivity = mean_fill
    productivity_difference = desired_productivity - actual_productivity
    if productivity_difference<0:
        productivity_difference = 0
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_range, y=actual_distribution_y, mode='lines', name=f'Actual Distribution for {shovel_of_interest} - {month_of_interest}', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_range, y=desired_distribution_y, mode='lines', name='Desired Distribution', line=dict(color='#00B7F1')))
    
    fig.add_trace(go.Scatter(x=[np.mean(actual_data), np.mean(actual_data)], y=[0, max(actual_distribution_y)], mode='lines', name='Actual Mean', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=[mean_fill, mean_fill], y=[0, max(desired_distribution_y)], mode='lines', name='Desired Mean', line=dict(color='#00B7F1', dash='dash')))

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
    
    st.plotly_chart(fig, use_container_width=True)

# Define a mapping of column names to standardize
# Define a mapping of column names to standardize
column_mapping = {
    'Tonnage': 'VIMS_tonnage',
    'Truck Factor': 'truck_factor_tonnage',
    'Shovel': 'shovel',
    'Shovel Type': 'shovel', 
    'Ton': 'VIMS_tonnage'
}



uploaded_files = st.sidebar.file_uploader("Upload your CSV Data Files", type=['csv'], accept_multiple_files=True)
if uploaded_files:
    dfs = []
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        
        # Extract month from filename and add it as a column
        month = uploaded_file.name.split('.')[0]  # Assuming the filename format is "month.csv"
        df['Month'] = extract_month_name(month)
        
        # Standardize column names based on the predefined mapping
        df = standardize_column_names(df, column_mapping)
        
        if 'shovel' not in df.columns:
            st.error("Uploaded data does not contain a 'shovel' column in any recognized format. Please upload a file with the correct format.")
            continue
        dfs.append(df)
    
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        # Define the month order
        
        month_order = ['Jan','January', 'Feb','February', 'Mar', 'March', 'Apr', 'April', 'May', 'Jun', 'June',
                       'Jul','July', 'Aug', 'August', 'Sep', 'September', 'Oct', 'October', 'Nov', 'November',
                       'Dec', 'December']

        # Create a temporary column with the month order
        df['Month_Order'] = df['Month'].apply(lambda x: month_order.index(x))

        # Sort the DataFrame by the temporary column
        df= df.sort_values(by='Month_Order')

        # Drop the temporary column
        df= df.drop(columns='Month_Order')    
        mean_fill = st.sidebar.slider("Select Mean Fill Percentage:", 98, 110, 104)
        target_std_dev = st.sidebar.slider("Select Standard Deviation:", 1, 10, 5)
        
        shovels_of_interest = ["EX201", "EX202", "EX203", "EX204", "EX208"]
        
        for shovel in shovels_of_interest:
            summary_data_all_months = []  # Initialize summary data for all months
            shovel_data = df[df['shovel']==shovel]
            # shovel_data = shovel_data.dropna()
            if not shovel_data.empty:
                generate_plot(shovel_data, shovel, target_std_dev=target_std_dev, mean_fill=mean_fill)
                for month in shovel_data['Month'].unique():
                    filtered_df = shovel_data[shovel_data['Month'] == month]
                    if not filtered_df.empty:
                        # Calculate summary data for the current month and shovel
                        total_trucks = len(filtered_df)
                        actual_material = filtered_df['VIMS_tonnage'].sum()
                        desired_material = total_trucks * np.mean(filtered_df['truck_factor_tonnage']) * mean_fill / 100
                        tonnage_increase = max(0, desired_material - actual_material)
                        actual_productivity = np.mean(filtered_df['Truck fill'])
                        desired_productivity = mean_fill
                        productivity_difference = desired_productivity - actual_productivity
                        if productivity_difference<0:
                            productivity_difference=0
                        # Append summary data for the current month and shovel to the list
                        summary_data_all_months.append({
                            'Month': month,
                            'Total Number of Trucks': total_trucks,
                            'Actual Material (Tonnes)': actual_material,
                            'Desired Material (Tonnes)': desired_material,
                            'Tonnage Increase': max(0, tonnage_increase),
                            'Productivity Increase (%)': productivity_difference*100
                        })

                # Display summary table for all months for the current shovel
                if summary_data_all_months:
                    summary_df_all_months = pd.DataFrame(summary_data_all_months)
                    # Calculate the sum for each numeric column
                    total_row = {
                        'Month': 'Total',
                        'Total Number of Trucks': summary_df_all_months['Total Number of Trucks'].sum(),
                        'Actual Material (Tonnes)': summary_df_all_months['Actual Material (Tonnes)'].sum(),
                        'Desired Material (Tonnes)': summary_df_all_months['Desired Material (Tonnes)'].sum(),
                        'Tonnage Increase': summary_df_all_months['Tonnage Increase'].sum(),
                        'Productivity Increase (%)': summary_df_all_months['Productivity Increase (%)'].sum() 
                    }
                    
                    # Append the total row to the DataFrame
                    # summary_df_all_months = summary_df_all_months.append(total_row, ignore_index=True)
                    summary_df_all_months = pd.concat([summary_df_all_months, pd.DataFrame([total_row])], ignore_index=True)
                    
                    # Optional: Format the numeric columns back to strings with appropriate formatting
                    summary_df_all_months['Total Number of Trucks'] = summary_df_all_months['Total Number of Trucks'].apply(lambda x: f'{x:.0f}')
                    summary_df_all_months['Actual Material (Tonnes)'] = summary_df_all_months['Actual Material (Tonnes)'].apply(lambda x: f'{x:.2e}')
                    summary_df_all_months['Desired Material (Tonnes)'] = summary_df_all_months['Desired Material (Tonnes)'].apply(lambda x: f'{x:.2e}')
                    summary_df_all_months['Tonnage Increase'] = summary_df_all_months['Tonnage Increase'].apply(lambda x: f'{x:.2e}')
                    summary_df_all_months['Productivity Increase (%)'] = summary_df_all_months['Productivity Increase (%)'].apply(lambda x: f'{x:.2f}%')

                    st.write(f"Summary for Shovel: {shovel}")
                    st.table(summary_df_all_months.style.set_table_styles([
                        {'selector': 'th',
                            'props': [('background-color', '#0077b6'),
                                    ('color', 'white'),
                                    ('text-align', 'center'),
                                    ('font-weight', 'bold'),
                                    ('font-size', '14px'),
                                    ('padding', '10px')]
                        },
                        {'selector': 'td',
                            'props': [('text-align', 'center'),
                                    ('font-size', '14px'),
                                    ('padding', '8px')]
                        },
                        {'selector': 'tr:nth-of-type(odd)',
                            'props': [('background-color', '#f0f0f0')]
                        },
                        {'selector': 'tr:nth-of-type(even)',
                            'props': [('background-color', 'white')]
                        }
                    ]))
