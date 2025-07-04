# Weather Forecasting with SFA-LSTM

This project implements a Spatial Feature Attention-based LSTM (SFA-LSTM) model for short-term temperature forecasting using hourly weather data. The implementation is inspired by the methodology presented in:

> Suleman, M. A. R., & Shridevi, S. (2022). Short-term weather forecasting using spatial feature attention based LSTM model. *IEEE Access*. <https://doi.org/10.1109/ACCESS.2022.3196381>

It uses the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs) and is structured for long-term reproducibility using [Poetry](https://python-poetry.org/).

## Features

- Multi-location weather data collection (Salem, Eugene, Corvallis, Oregon)
- Data cleaning, sequence generation, and model training pipeline
- Spatial feature attention for input variables
- Model saving and visualization of predictions
- Pipeline execution via a unified Python script with CLI options

## Directory Structure

```text
weather_data_project/
├── data/
│   ├── raw/            # Raw downloaded weather data
│   ├── interim/        # Cleaned but unsequenced data
│   ├── processed/      # Final sequences for modeling
├── models/             # Trained SFA-LSTM models
├── results/            # Evaluation plots and metrics
├── src/
│   └── weather_data_project/
│       ├── download_weather_openmeteo.py
│       ├── clean_data.py
│       ├── prepare_sequences.py
│       ├── combine_cities_sequences.py
│       ├── train_sfa_lstm.py
│       └── advanced_visualize_predictions.py
├── run_pipeline.py     # Unified script for running the full workflow
```

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/elpbatista/weather_data_project.git
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
