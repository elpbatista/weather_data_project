import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
sequence_length = 24  # default window size
target_column = "temperature"

# Paths
project_root = Path(__file__).resolve().parents[2]
interim_dir = project_root / "data" / "interim"
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# List all cleaned CSV files
csv_files = list(interim_dir.glob("*.csv"))

# Storage for combined data
combined_X, combined_y, city_labels = [], [], []

def create_sequences(df, seq_len, target_col):
    X, y = [], []
    feature_cols = [col for col in df.columns if col != target_col]

    for i in range(len(df) - seq_len):
        X_window = df.iloc[i:i+seq_len][feature_cols].values
        y_target = df.iloc[i+seq_len][target_col]
        X.append(X_window)
        y.append(y_target)

    return np.array(X), np.array(y)

# Process each city's data
for file in csv_files:
    city = file.stem.split("_")[0].lower()  # e.g., "salem"
    print(f"Processing: {city}")
    df = pd.read_csv(file, parse_dates=["time"])
    df.set_index("time", inplace=True)
    df.dropna(inplace=True)

    X, y = create_sequences(df, sequence_length, target_column)
    city_id = np.full((X.shape[0], 1), city)

    combined_X.append(X)
    combined_y.append(y)
    city_labels.extend(city_id)

# Combine all city data
X_all = np.concatenate(combined_X, axis=0)
y_all = np.concatenate(combined_y, axis=0)
city_all = np.array(city_labels).flatten()

# Flatten X for CSV
n_samples, n_steps, n_features = X_all.shape
flat_X = X_all.reshape(n_samples, n_steps * n_features)
columns = [
    f"{col}_t-{sequence_length - i - 1}"
    for i in range(sequence_length)
    for col in df.columns if col != target_column
]

X_df = pd.DataFrame(flat_X, columns=columns)
y_df = pd.DataFrame(y_all, columns=[target_column])
city_df = pd.DataFrame(city_all, columns=["city"])

# Combine and save
final_df = pd.concat([city_df, X_df, y_df], axis=1)
out_path = processed_dir / "combined_cities_weather_2013_2023.csv"
final_df.to_csv(out_path, index=False)
print(f"✅ Combined dataset saved to: {out_path}")
