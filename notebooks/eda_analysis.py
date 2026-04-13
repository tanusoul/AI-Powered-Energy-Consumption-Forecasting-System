# eda_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/energy_data.csv")

# Convert Datetime column
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Set Datetime as index
df.set_index('Datetime', inplace=True)

print("\nDataset Loaded Successfully!")

# Display basic information
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# Visualization 1: Energy Consumption Over Time
# -------------------------------
plt.figure(figsize=(14, 6))
plt.plot(df['Energy'], color='blue')
plt.title("Energy Consumption Over Time")
plt.xlabel("Datetime")
plt.ylabel("Energy (kW)")
plt.tight_layout()
plt.savefig("images/energy_over_time.png")
plt.show()

# -------------------------------
# Visualization 2: Daily Average Energy Consumption
# -------------------------------
daily_avg = df['Energy'].resample('D').mean()

plt.figure(figsize=(14, 6))
daily_avg.plot(color='green')
plt.title("Daily Average Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Energy (kW)")
plt.tight_layout()
plt.savefig("images/daily_average_energy.png")
plt.show()

# -------------------------------
# Visualization 3: Monthly Average Energy Consumption
# -------------------------------
monthly_avg = df['Energy'].resample('ME').mean()


plt.figure(figsize=(14, 6))
monthly_avg.plot(color='red')
plt.title("Monthly Average Energy Consumption")
plt.xlabel("Month")
plt.ylabel("Energy (kW)")
plt.tight_layout()
plt.savefig("images/monthly_average_energy.png")
plt.show()

# -------------------------------
# Visualization 4: Distribution Plot
# -------------------------------
plt.figure(figsize=(10, 5))
sns.histplot(df['Energy'], bins=50, kde=True)
plt.title("Distribution of Energy Consumption")
plt.xlabel("Energy (kW)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("images/energy_distribution.png")
plt.show()

print("\n✅ EDA Completed Successfully! Graphs saved in the 'images' folder.")