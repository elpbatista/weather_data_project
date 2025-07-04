import requests
import pandas as pd
from pathlib import Path

# --- Configuration ---
locations = {
    "salem": (44.94, -123.03),
    "eugene": (44.05, -123.09),
    "corvallis": (44.56, -123.26)
}

start_date = '2013-01-01'
end_date = '2023-12-31'
variables = [
    'temperature_2m',
    'dew_point_2m',
    'apparent_temperature',
    'relative_humidity_2m',
    'surface_pressure',
    'pressure_msl',
    'wind_speed_10m'
]

# --- Directory Setup ---
project_root = Path(__file__).resolve().parents[2]  # assumes script is inside src/weather_data_project/
data_dir = project_root / 'data'
raw_dir = data_dir / 'raw'
raw_dir.mkdir(parents=True, exist_ok=True)

# --- API Base URL ---
url = 'https://archive-api.open-meteo.com/v1/archive'

# --- Download and Save Loop ---
for city, (lat, lon) in locations.items():
    print(f"📡 Downloading data for {city.title()}...")

    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(variables),
        'temperature_unit': 'celsius',
        'wind_speed_unit': 'kmh',
        'timeformat': 'iso8601',
        'timezone': 'auto'
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()['hourly']

    # Create DataFrame
    df = pd.DataFrame({
        'time': data['time'],
        'temperature': data['temperature_2m'],
        'dew_point': data['dew_point_2m'],
        'wind_chill': data['apparent_temperature'],
        'relative_humidity': data['relative_humidity_2m'],
        'station_pressure': data['surface_pressure'],
        'sea_level_pressure': data['pressure_msl'],
        'wind_speed': data['wind_speed_10m'],
    })
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Save to data/raw/
    filename = raw_dir / f"{city}_weather_2013_2023.csv"
    df.to_csv(filename)
    print(f"✅ Saved: {filename}")