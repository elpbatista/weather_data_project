import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
sequence_length = 24
target_column = "temperature"
processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
models_dir = Path(__file__).resolve().parents[2] / "models"
output_dir = Path(__file__).resolve().parents[2] / "results"
output_dir.mkdir(parents=True, exist_ok=True)

def load_dataset(csv_file, has_city=False):
    df = pd.read_csv(csv_file)
    if has_city and 'city' in df.columns:
        df = df.drop(columns=['city'])
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    n_features = int(X.shape[1] / sequence_length)
    X = X.reshape((X.shape[0], sequence_length, n_features))
    return X, y

def plot_predictions(y_true, y_pred, title, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:200], label='Actual', linewidth=2)
    plt.plot(y_pred[:200], label='Predicted', linewidth=2)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Process each model and corresponding data file
for model_file in models_dir.glob("*.keras"):
    model_name = model_file.stem  # e.g., 'salem_weather_2013_2023_sfa_lstm'
    print(f"Evaluating model: {model_file.name}")

    if model_name == "combined_sfa_lstm":
        data_file = processed_dir / "combined_cities_weather_2013_2023.csv"
        has_city = True
    else:
        city_key = model_name.replace("_sfa_lstm", "")
        data_file = processed_dir / f"{city_key}.csv"
        has_city = False

    if not data_file.exists():
        print(f"Data file not found: {data_file.name}")
        continue

    X, y = load_dataset(data_file, has_city=has_city)
    model = load_model(model_file)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Metrics for {city_key}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}\n")

    plot_path = output_dir / f"{city_key}_prediction_plot.png"
    plot_predictions(y, y_pred.flatten(), f"Prediction vs Actual: {city_key}", plot_path)
    print(f"Saved plot to {plot_path}")
