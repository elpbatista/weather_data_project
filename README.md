# Weather Forecasting with SFA-LSTM

This project implements a Spatial Feature Attention-based LSTM (SFA-LSTM) model for short-term temperature forecasting using hourly weather data. The implementation is inspired by the methodology presented in:

> Suleman, M. A. R., & Shridevi, S. (2022). Short-term weather forecasting using spatial feature attention based LSTM model. *IEEE Access*. <https://doi.org/10.1109/ACCESS.2022.3196381>

It uses the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs) and is structured for long-term reproducibility using [Poetry](https://python-poetry.org/).

## Features

- Multi-location weather data collection (Salem, Eugene, Corvallis)
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

### 3. Usage

Run the entire pipeline:

```bash
python run_pipeline.py
```

Or skip specific steps using CLI flags:

```bash
python run_pipeline.py --skip-download --skip-train
```

#### Available Flags

- `--skip-download`      Skip weather data download
- `--skip-clean`         Skip data cleaning
- `--skip-prepare`       Skip sequence preparation
- `--skip-combine`       Skip combining datasets
- `--skip-train`         Skip model training
- `--skip-visualize`     Skip prediction visualization

## Citation

Please cite the original paper if you use this project in your work:

> Suleman, M. A. R., & Shridevi, S. (2022). Short-term weather forecasting using spatial feature attention based LSTM model. *IEEE Access*. <https://doi.org/10.1109/ACCESS.2022.3196381>

## License

MIT License. See `LICENSE` for details.
