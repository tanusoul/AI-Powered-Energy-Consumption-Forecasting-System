import pandas as pd

df = pd.read_csv("data/processed_energy_data.csv")
sample_df = df.tail(5000)
sample_df.to_csv("data/sample_energy_data.csv", index=False)