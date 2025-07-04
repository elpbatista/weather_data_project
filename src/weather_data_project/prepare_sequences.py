import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
sequence_length = 24  # default, can be changed
target_column = "temperature"

# Paths
project_root = Path(__file__).resolve().parents[2]
interim_dir = project_root / "data" / "interim"
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# List all cleaned files
csv_files = list(interim_dir.glob("*.csv"))

def create_sequences(df, seq_len, target_col):
    X, y = [], []
    feature_cols = [col for col in df.columns if col != target_col]

    for i in range(len(df) - seq_len):
        X_window = df.iloc[i:i+seq_len][feature_cols].values
        y_target = df.iloc[i+seq_len][target_col]
        X.append(X_window)
        y.append(y_target)

    return np.array(X), np.array(y)

# Process each file
for file in csv_files:
    print(f"Processing: {file.name}")
    df = pd.read_csv(file, parse_dates=["time"])
    df.set_index("time", inplace=True)
    df.dropna(inplace=True)

    # Create supervised sequences
    X, y = create_sequences(df, sequence_length, target_column)

    # Flatten X for CSV export (time-steps × features)
    n_samples, n_steps, n_features = X.shape
    flat_X = X.reshape(n_samples, n_steps * n_features)
    columns = [
        f"{col}_t-{sequence_length-i-1}"
        for i in range(sequence_length)
        for col in df.columns if col != target_column
    ]

    X_df = pd.DataFrame(flat_X, columns=columns)
    y_df = pd.DataFrame(y, columns=[target_column])

    # Combine X and y
    final_df = pd.concat([X_df, y_df], axis=1)

    # Save to processed/
    out_file = processed_dir / file.name
    final_df.to_csv(out_file, index=False)
    print(f"Saved processed sequence data to: {out_file}")
