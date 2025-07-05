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

#### Run the Entire Pipeline

```bash
poetry run python run_pipeline.py
```

#### Skip Specific Steps

You can skip any step using the `--skip` flag followed by one or more step names:

```bash
poetry run python run_pipeline.py --skip download train
```

#### Run Only Selected Steps

To run only specific steps, use the `--only` flag followed by one or more step names:

```bash
poetry run python run_pipeline.py --only download clean train
```

**Available step names for `--only` and `--skip`:**

- `download`           Download weather data
- `clean`              Clean raw data
- `prepare`            Prepare sequences
- `combine`            Combine datasets
- `train`              Train SFA-LSTM models
- `visualize`          Visualize predictions
- `advanced-visualize` Advanced prediction visualization

**All available flags**:

- `--skip`   Skip the specified steps (see above)
- `--only`   Run only the specified steps (see above)

## Citation

Please cite the original paper if you use this project in your work:

> Suleman, M. A. R., & Shridevi, S. (2022). Short-term weather forecasting using spatial feature attention based LSTM model. *IEEE Access*. <https://doi.org/10.1109/ACCESS.2022.3196381>

## License

MIT License. See `LICENSE` for details.
