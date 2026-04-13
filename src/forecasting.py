# src/forecasting.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Create necessary directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("images", exist_ok=True)


def load_model(model_path):
    """Load the trained model."""
    print("Loading trained model...")
    return joblib.load(model_path)


def load_data(file_path):
    """Load the processed dataset."""
    print("Loading processed dataset...")
    df = pd.read_csv(file_path, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    return df


def forecast_future(model, df, steps=24):
    """
    Forecast future energy consumption.
    Predicts the next 'steps' hours.
    """
    print(f"Forecasting next {steps} hours...")
    last_row = df.iloc[-1:].copy()
    predictions = []
    future_dates = []

    for i in range(steps):
        # Predict next value
        pred = model.predict(last_row)[0]
        predictions.append(pred)

        # Generate next timestamp
        next_time = last_row.index[0] + pd.Timedelta(hours=1)
        future_dates.append(next_time)

        # Create new row
        new_row = last_row.copy()
        new_row.index = [next_time]
        new_row['Energy'] = pred

        # Update time-based features
        new_row['hour'] = next_time.hour
        new_row['day'] = next_time.day
        new_row['month'] = next_time.month
        new_row['day_of_week'] = next_time.dayofweek
        new_row['week_of_year'] = next_time.isocalendar().week
        new_row['is_weekend'] = int(next_time.dayofweek >= 5)

        # Update lag features
        new_row['lag_1'] = pred
        new_row['lag_24'] = pred
        new_row['lag_168'] = pred
        new_row['rolling_mean_24'] = pred
        new_row['rolling_mean_168'] = pred

        # Keep only feature columns
        last_row = new_row.drop(columns=['Energy'])

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Datetime': future_dates,
        'Predicted_Energy': predictions
    })

    forecast_df.set_index('Datetime', inplace=True)
    return forecast_df


def plot_forecast(df, forecast_df):
    """Plot historical and forecasted energy."""
    plt.figure(figsize=(14, 6))

    # Plot historical data
    plt.plot(df.index[-200:], df['Energy'].tail(200),
             label="Historical Energy", color="blue")

    # Plot forecast
    plt.plot(forecast_df.index, forecast_df['Predicted_Energy'],
             label="Forecasted Energy", linestyle="--", color="red")

    plt.title("Energy Consumption Forecast")
    plt.xlabel("Datetime")
    plt.ylabel("Energy (kW)")
    plt.legend()
    plt.tight_layout()

    plt.savefig("images/energy_forecast.png")
    plt.show()


def save_forecast(forecast_df):
    """Save forecast results."""
    output_path = "outputs/future_energy_forecast.csv"
    forecast_df.to_csv(output_path)
    print(f"Forecast saved to {output_path}")


def main():
    # File paths
    model_path = "models/random_forest_model.pkl"
    data_path = "data/processed_energy_data.csv"

    # Load model and data
    model = load_model(model_path)
    df = load_data(data_path)

    # Separate features and target
    features = df.drop(columns=['Energy'])

    # Generate forecast
    forecast_df = forecast_future(model, features, steps=24)

    # Plot and save results
    plot_forecast(df, forecast_df)
    save_forecast(forecast_df)

    print("\n✅ Forecasting completed successfully!")


if __name__ == "__main__":
    main()