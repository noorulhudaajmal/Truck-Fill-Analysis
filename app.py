import streamlit as st
import pandas as pd
import numpy as np
from utils import standardize_column_names, preprocessdata, generate_plot, filter_data

# ------------------------------ Page Configuration------------------------------
st.set_page_config(page_title="Truck Fill Analysis", page_icon="📊", layout="wide")

# ----------------------------------- Page Styling ------------------------------
st.markdown("""
        <style>
               .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               [data-testid=stSidebarUserContent] {
                    padding-top: 4rem;
                }
        </style>
        """, unsafe_allow_html=True)


# Function to upload files and set parameters
def upload_files_and_set_params():
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your CSV Data Files", type=['csv'], accept_multiple_files=True)
        mean_fill = st.slider("Select Mean Fill Percentage:", 98, 110, 104)
        target_std_dev = st.slider("Select Standard Deviation:", 1, 10, 5)
    return uploaded_files, mean_fill, target_std_dev


# Function to process uploaded files
def process_uploaded_files(uploaded_files):
    dfs = []
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        df = standardize_column_names(df)
        if 'shovel' not in df.columns:
            st.error("Uploaded data does not contain a 'shovel' column in any recognized format. "
                     "Please upload a file with the correct format.")
            continue
        dfs.append(df)
    return dfs


# Function to preprocess and filter data
def preprocess_and_filter_data(dfs, mean_fill):
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        columns_of_interest = ['shovel', 'time_full', 'VIMS_tonnage', 'truck_factor_tonnage']
        df = df.dropna(subset=columns_of_interest)
        df = preprocessdata(df)
        df = df.sort_values(by='Month Order')
        df = df.drop(columns='Month Order')
        df = filter_data_based_on_user_selection(df)
        return df
    return None


# Function to filter data based on user selection
def filter_data_based_on_user_selection(df):
    # filters_row = st.columns((2, 2, 1))
    all_shovels = sorted(df['shovel'].unique())
    shovel = st.sidebar.selectbox("Select Shovel:", all_shovels)

    avb_materials = list(df['Material'].unique()) + ["All"]
    material = st.sidebar.multiselect(label="Select material", options=avb_materials, default="All", placeholder="All")
    avb_dump = list(df['Dump'].unique()) + ["All"]
    dump = st.sidebar.multiselect(label="Select Dump", options=avb_dump, default="All", placeholder="All")
    all_shifts = ['7am-7pm', '7pm-7am', 'All']
    shift = st.sidebar.multiselect(label="Choose shift", options=all_shifts, default="All", placeholder="All")

    if (not material) or (material == ["All"]) or ("All" in material):
        material = df['Material'].unique()
    if (not dump) or (dump == ["All"]) or ("All" in dump):
        dump = df['Dump'].unique()
    if (not shift) or (shift == ["All"]) or ("All" in shift):
        shift = ['7am-7pm', '7pm-7am']
    df = filter_data(df, material, dump, shovel, shift)

    return df


# Function to generate plots and summary tables
def generate_plots_and_summaries(df, mean_fill, target_std_dev):
    if not df.empty:
        fig = generate_plot(df, target_std_dev=target_std_dev, mean_fill=mean_fill)
        st.plotly_chart(fig, use_container_width=True)
        summary_df_all_months = generate_summary_data(df, mean_fill)
        if not summary_df_all_months.empty:
            shovel_name = df['shovel'].iloc[0]
            display_summary_table(summary_df_all_months, shovel_name)
    else:
        st.info("No data available")


# Function to generate summary data for all months for a specific shovel
def generate_summary_data(shovel_data, mean_fill):
    summary_data_all_months = []
    for month in shovel_data['Month'].unique():
        filtered_df = shovel_data[shovel_data['Month'] == month]
        if not filtered_df.empty:
            summary_data = calculate_summary_data(filtered_df, month, mean_fill)
            summary_data_all_months.append(summary_data)
    if summary_data_all_months:
        summary_df_all_months = pd.DataFrame(summary_data_all_months)
        summary_df_all_months = append_total_row(summary_df_all_months)
        return summary_df_all_months
    return pd.DataFrame()


# Function to calculate summary data for a specific month and shovel
def calculate_summary_data(filtered_df, month, mean_fill):
    total_trucks = len(filtered_df)
    actual_material = filtered_df['VIMS_tonnage'].sum()
    desired_material = total_trucks * np.mean(filtered_df['truck_factor_tonnage']) * mean_fill / 100
    tonnage_increase = max(0, desired_material - actual_material)
    actual_productivity = np.mean(filtered_df['Truck fill'])
    desired_productivity = mean_fill
    productivity_difference = max(0, desired_productivity - actual_productivity)
    return {
        'Month': month,
        'Total Number of Trucks': total_trucks,
        'Actual Material (Tonnes)': actual_material,
        'Desired Material (Tonnes)': desired_material,
        'Tonnage Increase': tonnage_increase,
        'Productivity Increase (%)': productivity_difference
    }


# Function to append a total row to the summary DataFrame
def append_total_row(summary_df_all_months):
    total_row = {
        'Month': 'Total',
        'Total Number of Trucks': summary_df_all_months['Total Number of Trucks'].sum(),
        'Actual Material (Tonnes)': summary_df_all_months['Actual Material (Tonnes)'].sum(),
        'Desired Material (Tonnes)': summary_df_all_months['Desired Material (Tonnes)'].sum(),
        'Tonnage Increase': summary_df_all_months['Tonnage Increase'].sum(),
        'Productivity Increase (%)': summary_df_all_months['Productivity Increase (%)'].mean()
    }
    summary_df_all_months = pd.concat([summary_df_all_months, pd.DataFrame([total_row])], ignore_index=True)
    return summary_df_all_months


# Function to display the summary table for a specific shovel
def display_summary_table(summary_df, shovel):
    st.write(f"Summary for Shovel: {shovel}")
    # Format the numeric columns back to strings with appropriate formatting
    summary_df['Total Number of Trucks'] = summary_df['Total Number of Trucks'].apply(lambda x: f'{x:.0f}')
    summary_df['Actual Material (Tonnes)'] = summary_df['Actual Material (Tonnes)'].apply(lambda x: f'{x:.2e}')
    summary_df['Desired Material (Tonnes)'] = summary_df['Desired Material (Tonnes)'].apply(lambda x: f'{x:.2e}')
    summary_df['Tonnage Increase'] = summary_df['Tonnage Increase'].apply(lambda x: f'{x:.2e}')
    summary_df['Productivity Increase (%)'] = summary_df['Productivity Increase (%)'].apply(lambda x: f'{x:.2f}%')

    st.table(summary_df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#0077b6'), ('color', 'white'), ('text-align', 'center'),
                                     ('font-weight', 'bold'), ('font-size', '14px'), ('padding', '10px')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px'), ('padding', '8px')]},
        {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', '#f0f0f0')]},
        {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', 'white')]}
    ]))


# Main function to run the Streamlit app
def main():
    st.title("PLM Truck Fill Analysis")
    uploaded_files, mean_fill, target_std_dev = upload_files_and_set_params()
    if uploaded_files:
        dfs = process_uploaded_files(uploaded_files)
        df = preprocess_and_filter_data(dfs, mean_fill)
        if df is not None:
            generate_plots_and_summaries(df, mean_fill, target_std_dev)


if __name__ == "__main__":
    main()
