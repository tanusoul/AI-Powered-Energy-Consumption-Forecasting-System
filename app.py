import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="AI Energy Forecasting Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ------------------------------
# Custom Styling
# ------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
            color: white;
        }
        .stMetric {
            background-color: #1c1f26;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI-Powered Energy Consumption Forecasting System")
st.markdown("An advanced AI solution for smart cities and sustainable energy management.")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Linear Regression"]
)

forecast_steps = st.sidebar.slider(
    "Forecast Horizon (Hours)",
    min_value=1,
    max_value=168,
    value=24
)

# ------------------------------
# File Paths
# ------------------------------
DATA_PATH = "data/processed_energy_data.csv"
MODEL_PATHS = {
    "Random Forest": "models/random_forest_model.pkl",
    "Linear Regression": "models/linear_regression_model.pkl"
}

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
    df.set_index("Datetime", inplace=True)
    return df

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# ------------------------------
# Forecast Function
# ------------------------------
def forecast_energy(model, df, steps):
    features = df.drop(columns=["Energy"])
    last_row = features.iloc[-1:].copy()
    last_datetime = df.index[-1]

    predictions = []
    timestamps = []

    for i in range(steps):
        pred = model.predict(last_row)[0]
        next_time = last_datetime + pd.Timedelta(hours=1)

        predictions.append(pred)
        timestamps.append(next_time)

        new_row = last_row.copy()
        new_row.index = [next_time]

        new_row["hour"] = next_time.hour
        new_row["day"] = next_time.day
        new_row["month"] = next_time.month
        new_row["day_of_week"] = next_time.dayofweek
        new_row["week_of_year"] = next_time.isocalendar().week
        new_row["is_weekend"] = int(next_time.dayofweek >= 5)
        new_row["lag_1"] = pred
        new_row["lag_24"] = pred
        new_row["lag_168"] = pred
        new_row["rolling_mean_24"] = pred
        new_row["rolling_mean_168"] = pred

        last_row = new_row
        last_datetime = next_time

    forecast_df = pd.DataFrame({
        "Datetime": timestamps,
        "Predicted Energy": predictions
    }).set_index("Datetime")

    return forecast_df

# ------------------------------
# Load Resources
# ------------------------------
if not os.path.exists(DATA_PATH):
    st.error("Processed dataset not found. Please run feature_engineering.py.")
    st.stop()

df = load_data()
model = load_model(MODEL_PATHS[model_choice])

# ------------------------------
# Tabs Layout
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "📈 Visualizations",
    "🔮 Forecasting",
    "📉 Model Insights"
])

# ------------------------------
# Tab 1: Data Overview
# ------------------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Average Energy", f"{df['Energy'].mean():.2f} kW")
    col3.metric("Max Energy", f"{df['Energy'].max():.2f} kW")

# ------------------------------
# Tab 2: Visualizations
# ------------------------------
with tab2:
    st.subheader("Energy Consumption Over Time")
    fig = px.line(df.tail(1000), y="Energy",
                  title="Energy Consumption Trend")
    st.plotly_chart(fig, width="stretch")

# ------------------------------
# Tab 3: Forecasting
# ------------------------------
with tab3:
    st.subheader("Future Energy Forecasting")

    if st.button("Generate Forecast"):
        forecast_df = forecast_energy(model, df, forecast_steps)

        st.dataframe(forecast_df)

        combined = pd.concat([
            df[['Energy']].tail(200),
            forecast_df.rename(columns={"Predicted Energy": "Energy"})
        ])

        fig = px.line(
            combined,
            y="Energy",
            title="Historical vs Forecasted Energy Consumption"
        )

        st.plotly_chart(fig, use_container_width=True)

        csv = forecast_df.to_csv().encode("utf-8")
        st.download_button(
            "📥 Download Forecast CSV",
            csv,
            "future_energy_forecast.csv",
            "text/csv"
        )

# ------------------------------
# Tab 4: Model Insights
# ------------------------------
with tab4:
    st.subheader("Model Information")

    st.write(f"**Selected Model:** {model_choice}")

    if os.path.exists("outputs/model_results.csv"):
        results = pd.read_csv("outputs/model_results.csv")
        st.dataframe(results)

        fig = px.bar(
            results,
            x="Model",
            y="R2 Score",
            title="Model Performance Comparison",
            color="Model"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model results file not found.")

st.success("✅ Dashboard Loaded Successfully!")