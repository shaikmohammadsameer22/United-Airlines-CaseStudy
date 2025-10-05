import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# Page configuration
st.set_page_config(page_title="United Airlines Flight Difficulty Dashboard", layout="wide")

st.title("United Airlines Flight Difficulty Dashboard")
st.markdown("""
This dashboard analyzes flight difficulty based on operational metrics such as:
- Load Factor  
- Special Service Requests (SSR)  
- Departure Delays  
- Ground Time Buffer  
- Baggage Transfer Ratios  
""")





# Get folder of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to your data folder inside the repo
base_path = os.path.join(current_dir, "../casestudy_data")  

# Read CSVs
pnr_remark = pd.read_csv(os.path.join(base_path, "PNR Remark Level Data.csv"))
pnr_flight = pd.read_csv(os.path.join(base_path, "PNR Flight Level Data.csv"))
flight = pd.read_csv(os.path.join(base_path, "Flight Level Data.csv"))
bags = pd.read_csv(os.path.join(base_path, "Bag+Level+Data.csv"))

# Filter for United Airlines
flight = flight[flight['company_id'] == 'UA']
bags = bags[bags['company_id'] == 'UA']

# Merge passenger-level data
merge1 = pd.merge(
    pnr_flight,
    pnr_remark,
    on=['record_locator', 'pnr_creation_date', 'flight_number'],
    how='left'
)

merge1['scheduled_departure_date_local'] = pd.to_datetime(
    merge1['scheduled_departure_date_local'], errors='coerce'
).dt.date.astype(str)

flight['scheduled_departure_datetime_local'] = pd.to_datetime(
    flight['scheduled_departure_datetime_local'], errors='coerce'
)
flight['actual_departure_datetime_local'] = pd.to_datetime(
    flight['actual_departure_datetime_local'], errors='coerce'
)
flight['scheduled_departure_date_local'] = flight['scheduled_departure_datetime_local'].dt.date.astype(str)

# Combine datasets
final_df = pd.merge(
    merge1,
    flight[['flight_number','scheduled_departure_date_local',
            'scheduled_departure_datetime_local','actual_departure_datetime_local',
            'total_seats','scheduled_ground_time_minutes','minimum_turn_minutes']],
    on=['flight_number','scheduled_departure_date_local'],
    how='left'
)

# Compute delay and load factor
final_df['departure_delay_minutes'] = (
    final_df['actual_departure_datetime_local'] - final_df['scheduled_departure_datetime_local']
).dt.total_seconds() / 60
final_df['departure_delay_minutes'] = final_df['departure_delay_minutes'].clip(lower=0)

final_df['load_factor'] = np.where(
    final_df['total_seats'] > 0,
    final_df['total_pax'] / final_df['total_seats'],
    np.nan
)
final_df['special_service_request'] = final_df['special_service_request'].fillna(0)

# Flight summary aggregation
flight_summary = final_df.groupby(
    ['flight_number','scheduled_departure_date_local']
).agg(
    ssr_count=('special_service_request', lambda x: (x != 0).sum()),
    total_passengers=('total_pax', 'sum'),
    total_seats=('total_seats', 'first'),
    avg_load_factor=('load_factor', 'mean'),
    avg_delay=('departure_delay_minutes', 'mean'),
    ground_time=('scheduled_ground_time_minutes','first'),
    min_turn=('minimum_turn_minutes','first')
).reset_index()

flight_summary['ground_buffer'] = flight_summary['ground_time'] - flight_summary['min_turn']

# Baggage ratios
bags['bag_type'] = bags['bag_type'].str.strip().str.lower()
bag_counts = bags.groupby(['flight_number','bag_type']).size().unstack(fill_value=0)

for col in ['origin','transfer','hot transfer']:
    if col not in bag_counts.columns:
        bag_counts[col] = 0

bag_counts['total_transfer'] = bag_counts['transfer'] + bag_counts['hot transfer']
bag_counts['checked'] = bag_counts['origin']

bag_counts['transfer_to_checked_ratio'] = np.where(
    bag_counts['checked'] == 0,
    np.nan,
    bag_counts['total_transfer'] / bag_counts['checked']
)

bag_counts = bag_counts.reset_index()

flight_summary = pd.merge(
    flight_summary,
    bag_counts[['flight_number','transfer_to_checked_ratio']],
    on='flight_number',
    how='left'
)

# Handle missing values
for col in ['avg_load_factor','ssr_count','avg_delay','ground_buffer','transfer_to_checked_ratio']:
    flight_summary[col] = flight_summary[col].fillna(flight_summary[col].median())

# Normalization
flight_summary['load_norm']  = (flight_summary['avg_load_factor'] - flight_summary['avg_load_factor'].min()) / (flight_summary['avg_load_factor'].max() - flight_summary['avg_load_factor'].min())
flight_summary['ssr_norm']   = (flight_summary['ssr_count'] - flight_summary['ssr_count'].min()) / (flight_summary['ssr_count'].max() - flight_summary['ssr_count'].min())
flight_summary['delay_norm'] = (flight_summary['avg_delay'] - flight_summary['avg_delay'].min()) / (flight_summary['avg_delay'].max() - flight_summary['avg_delay'].min())
flight_summary['ground_norm']= 1 - ((flight_summary['ground_buffer'] - flight_summary['ground_buffer'].min()) / (flight_summary['ground_buffer'].max() - flight_summary['ground_buffer'].min()))
flight_summary['bag_norm']   = (flight_summary['transfer_to_checked_ratio'] - flight_summary['transfer_to_checked_ratio'].min()) / (flight_summary['transfer_to_checked_ratio'].max() - flight_summary['transfer_to_checked_ratio'].min())

# Calculate factor weights based on correlation with delay
corrs = {
    'load_norm': abs(flight_summary['load_norm'].corr(flight_summary['avg_delay'])),
    'ssr_norm': abs(flight_summary['ssr_norm'].corr(flight_summary['avg_delay'])),
    'delay_norm': abs(flight_summary['delay_norm'].corr(flight_summary['avg_delay'])),
    'ground_norm': abs(flight_summary['ground_norm'].corr(flight_summary['avg_delay'])),
    'bag_norm': abs(flight_summary['bag_norm'].corr(flight_summary['avg_delay']))
}
corr_df = pd.DataFrame(list(corrs.items()), columns=['factor','corr_value'])
corr_df['weight'] = corr_df['corr_value'] / corr_df['corr_value'].sum()
weights = dict(zip(corr_df['factor'], corr_df['weight']))

# Difficulty score
flight_summary['difficulty_score'] = (
    weights['load_norm']  * flight_summary['load_norm'] +
    weights['ssr_norm']   * flight_summary['ssr_norm'] +
    weights['delay_norm'] * flight_summary['delay_norm'] +
    weights['ground_norm']* flight_summary['ground_norm'] +
    weights['bag_norm']   * flight_summary['bag_norm']
)

flight_summary['difficulty_rank'] = flight_summary.groupby('scheduled_departure_date_local')['difficulty_score'].rank(ascending=False)
flight_summary['difficulty_category'] = pd.qcut(
    flight_summary['difficulty_rank'],
    q=3,
    labels=['Easy','Medium','Difficult']
)

# Sidebar filters
st.sidebar.header("Filters")
date_list = sorted(flight_summary['scheduled_departure_date_local'].unique())
selected_date = st.sidebar.selectbox("Select Date", date_list)
filtered = flight_summary[flight_summary['scheduled_departure_date_local'] == selected_date]

# KPI cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Delay (min)", f"{filtered['avg_delay'].mean():.1f}")
col2.metric("Avg Load Factor", f"{filtered['avg_load_factor'].mean()*100:.1f}%")
col3.metric("Total Passengers", int(filtered['total_passengers'].sum()))
col4.metric("Difficult Flights", (filtered['difficulty_category'] == 'Difficult').sum())

# Visualizations
st.markdown("### Flight Difficulty Overview")

fig1 = px.bar(
    filtered.sort_values("difficulty_score", ascending=False),
    x="flight_number",
    y="difficulty_score",
    color="difficulty_category",
    title=f"Flight Difficulty Scores on {selected_date}",
    color_discrete_map={'Easy':'green','Medium':'orange','Difficult':'red'}
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(
    filtered,
    x="avg_delay", y="difficulty_score",
    color="difficulty_category",
    size="ssr_count",
    hover_data=["flight_number"],
    title="Average Delay vs Difficulty Score"
)
st.plotly_chart(fig2, use_container_width=True)

# Detailed data and weights
st.markdown("### Detailed Flight Data")
st.dataframe(filtered[['flight_number','avg_delay','avg_load_factor','ssr_count',
                       'ground_buffer','transfer_to_checked_ratio',
                       'difficulty_score','difficulty_category']])

st.markdown("### Factor Weights Used in Scoring")
st.dataframe(corr_df)

st.success("Dashboard Loaded Successfully")
