 
import pandas as pd

# File paths
input_file = "data/household_power_consumption.txt"
output_file = "data/energy_data.csv"

print("Loading dataset... This may take a few moments.")

try:
    # Load dataset
    df = pd.read_csv(
        input_file,
        sep=';',               # Dataset uses semicolon as separator
        low_memory=False,
        na_values=['?']        # Replace '?' with NaN
    )

    print("Dataset loaded successfully!")

    # Combine Date and Time into a single Datetime column
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S'
    )

    # Select relevant columns
    df = df[['Datetime', 'Global_active_power']]

    # Rename columns
    df.rename(columns={'Global_active_power': 'Energy'}, inplace=True)

    # Convert Energy column to numeric
    df['Energy'] = pd.to_numeric(df['Energy'], errors='coerce')

    # Remove missing values
    df.dropna(inplace=True)

    # Sort by date and time
    df.sort_values(by='Datetime', inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Save to CSV
    df.to_csv(output_file, index=False)

    print("\n✅ Conversion Successful!")
    print(f"Saved as: {output_file}")
    print("\nPreview of the dataset:")
    print(df.head())
    print("\nDataset Shape:", df.shape)

except FileNotFoundError:
    print("\n❌ Error: Dataset file not found.")
    print("Ensure the file is placed in the 'data' folder with the correct name.")