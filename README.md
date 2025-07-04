# Weather Data Project

This project automates the download, organization, and processing of hourly weather data for three Oregon cities in the Willamette Valley: Salem, Eugene, and Corvallis.

It uses the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs) and is structured for long-term reproducibility using [Poetry](https://python-poetry.org/).

## Project Structure

```text
weather_data_project/
├── data/
│   ├── raw/         # Raw data directly from Open-Meteo
│   ├── interim/     # Cleaned/merged, not final
│   └── processed/   # Final datasets ready for modeling
├── pyproject.toml   # Poetry project and dependencies
├── README.md
├── src/
│   └── weather_data_project/
│       └── download_weather_openmeteo.py
└── tests/
```

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/weather_data_project.git
cd weather_data_project
```

### 2. Install Dependencies with Poetry

If Poetry is not installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then install the project dependencies:

```bash
poetry install
```

### 3. Run the Weather Data Download Script

```bash
poetry run python src/weather_data_project/download_weather_openmeteo.py
```

This will save hourly weather data (2013–2023) to:

```text
data/raw/salem_weather_2013_2023.csv
data/raw/eugene_weather_2013_2023.csv
data/raw/corvallis_weather_2013_2023.csv
```

## Features Downloaded

- Temperature (`temperature_2m`)
- Dew Point (`dew_point_2m`)
- Wind Chill / Apparent Temperature (`apparent_temperature`)
- Relative Humidity (`relative_humidity_2m`)
- Station Pressure (`surface_pressure`)
- Sea Level Pressure (`pressure_msl`)
- Wind Speed (`wind_speed_10m`)

## Future Development Ideas

- Clean and validate raw data to `data/interim/`
- Generate analysis-ready data in `data/processed/`
- Forecast temperature using models such as SFA-LSTM
- Visualize time series trends across locations
- Add CLI options, Makefile, or orchestration tools

## License

MIT License. See `LICENSE` for details.
