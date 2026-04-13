import pandas as pd
import os


def load_data(file_path):
    """Load and prepare the dataset."""
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    return df


def create_time_features(df):
    """Create time-based features."""
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df


def create_lag_features(df):
    """Create lag-based features."""
    df['lag_1'] = df['Energy'].shift(1)
    df['lag_24'] = df['Energy'].shift(24)
    df['lag_168'] = df['Energy'].shift(168)  # Weekly lag
    return df


def create_rolling_features(df):
    """Create rolling mean features."""
    df['rolling_mean_24'] = df['Energy'].rolling(window=24).mean()
    df['rolling_mean_168'] = df['Energy'].rolling(window=168).mean()
    return df


def preprocess_data(input_path, output_path):
    """Complete feature engineering pipeline."""
    print("Loading dataset...")
    df = load_data(input_path)

    print("Creating time-based features...")
    df = create_time_features(df)

    print("Creating lag features...")
    df = create_lag_features(df)

    print("Creating rolling mean features...")
    df = create_rolling_features(df)

    print("Dropping missing values...")
    df.dropna(inplace=True)

    print("Saving processed dataset...")
    df.to_csv(output_path)

    print("✅ Feature Engineering Completed!")
    print("Processed dataset saved at:", output_path)
    print("\nDataset Shape:", df.shape)

    return df


if __name__ == "__main__":
    input_file = "data/energy_data.csv"
    output_file = "data/processed_energy_data.csv"
    preprocess_data(input_file, output_file)