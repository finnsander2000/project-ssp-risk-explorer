df_dummy_path = "D:\\GitHub\\project-ssp-risk-explorer\\data\\raw\\dummy\\dashboard_ready_dummy_ssp_poverty_multiline.csv"


import streamlit as st
import pandas as pd


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

df = load_data(df_dummy_path)
st.title("SSP Poverty Multiline Dashboard")
st.write("This is a dummy dashboard for SSP Poverty Multiline data.")
st.dataframe(df)
st.line_chart(df, x='Year', y=['Poverty Rate SSP1', 'Poverty Rate SSP2', 'Poverty Rate SSP3'])  
# Load the data
# Display the data in a table
# Create a line chart to visualize poverty rates over time for different SSPs




