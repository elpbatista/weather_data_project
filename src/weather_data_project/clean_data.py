import pandas as pd
from pathlib import Path

# Define input/output directories
project_root = Path(__file__).resolve().parents[2]
raw_dir = project_root / "data" / "raw"
interim_dir = project_root / "data" / "interim"
interim_dir.mkdir(parents=True, exist_ok=True)

# List CSV files in raw data directory
csv_files = list(raw_dir.glob("*.csv"))

# Columns to smooth (everything except temperature)
columns_to_smooth = [
    "dew_point",
    "wind_chill",
    "relative_humidity",
    "station_pressure",
    "sea_level_pressure",
    "wind_speed"
]

rolling_window = 5  # hours

def clean_dataframe(df):
    # Ensure datetime index
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

    # Interpolate missing values
    df = df.interpolate(method="linear", limit_direction="forward")

    # Apply rolling average to selected columns only
    for col in columns_to_smooth:
        if col in df.columns:
            df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean()

    return df

# Process each raw CSV
for file in csv_files:
    print(f"Processing: {file.name}")
    df = pd.read_csv(file)

    # Clean and smooth
    df_clean = clean_dataframe(df)

    # Save to interim/
    out_file = interim_dir / file.name
    df_clean.to_csv(out_file)
    print(f"Saved cleaned file to: {out_file}")
