# src/model_training.py

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("images", exist_ok=True)


def load_data(file_path):
    """Load processed dataset."""
    df = pd.read_csv(file_path, parse_dates=['Datetime'])
    return df


def prepare_features(df):
    """Prepare features and target variable."""
    X = df.drop(columns=['Energy', 'Datetime'])
    y = df['Energy']
    return X, y


def split_data(X, y):
    """Split data into training and testing sets."""
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test


def evaluate_model(model_name, y_test, y_pred):
    """Evaluate model performance."""
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 {model_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return rmse, r2


def save_metrics(results):
    """Save evaluation metrics to CSV."""
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/model_results.csv", index=False)


def plot_predictions(y_test, predictions, model_name):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:500], label="Actual", linewidth=2)
    plt.plot(predictions[:500], label="Predicted", linestyle="--")
    plt.title(f"{model_name} - Actual vs Predicted Energy Consumption")
    plt.xlabel("Time")
    plt.ylabel("Energy (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{model_name.lower().replace(' ', '_')}_predictions.png")
    plt.show()


def train_models(file_path):
    """Train and evaluate ML models."""
    print("Loading processed dataset...")
    df = load_data(file_path)

    print("Preparing features and target...")
    X, y = prepare_features(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    results = []

    # ---------------------------
    # Linear Regression
    # ---------------------------
    print("\nTraining Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    lr_predictions = lr_model.predict(X_test)
    lr_rmse, lr_r2 = evaluate_model(
        "Linear Regression", y_test, lr_predictions
    )

    joblib.dump(lr_model, "models/linear_regression_model.pkl")
    plot_predictions(y_test, lr_predictions, "Linear Regression")

    results.append({
        "Model": "Linear Regression",
        "RMSE": lr_rmse,
        "R2 Score": lr_r2
    })

    # ---------------------------
    # Random Forest
    # ---------------------------
    print("\nTraining Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)
    rf_rmse, rf_r2 = evaluate_model(
        "Random Forest Regressor", y_test, rf_predictions
    )

    joblib.dump(rf_model, "models/random_forest_model.pkl")
    plot_predictions(y_test, rf_predictions, "Random Forest")

    results.append({
        "Model": "Random Forest Regressor",
        "RMSE": rf_rmse,
        "R2 Score": rf_r2
    })

    # Save evaluation results
    save_metrics(results)
    print("\n✅ Model training completed successfully!")
    print("📁 Models saved in the 'models' folder.")
    print("📊 Results saved in the 'outputs' folder.")


if __name__ == "__main__":
    train_models("data/processed_energy_data.csv")