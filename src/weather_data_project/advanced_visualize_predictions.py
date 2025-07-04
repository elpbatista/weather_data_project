import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
sequence_length = 24
target_column = "temperature"
processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
models_dir = Path(__file__).resolve().parents[2] / "models"
output_dir = Path(__file__).resolve().parents[2] / "results"
output_dir.mkdir(parents=True, exist_ok=True)

all_metrics = []
all_predictions = {}

def load_dataset(csv_file, has_city=False):
    df = pd.read_csv(csv_file)
    if has_city and 'city' in df.columns:
        df = df.drop(columns=['city'])
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    n_features = int(X.shape[1] / sequence_length)
    X = X.reshape((X.shape[0], sequence_length, n_features))
    return X, y

# Process each model
for model_file in models_dir.glob("*.keras"):
    model_name = model_file.stem
    print(f"Evaluating model: {model_file.name}")

    if model_name == "combined_sfa_lstm":
        data_file = processed_dir / "combined_cities_weather_2013_2023.csv"
        has_city = True
        label = "Combined"
    else:
        city_key = model_name.replace("_sfa_lstm", "")
        data_file = processed_dir / f"{city_key}.csv"
        has_city = False
        label = city_key.split("_")[0].capitalize()

    if not data_file.exists():
        print(f"Data file not found: {data_file.name}")
        continue

    X, y = load_dataset(data_file, has_city=has_city)
    model = load_model(model_file)
    y_pred = model.predict(X).flatten()

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    residuals = y - y_pred

    print(f"{label} | MSE: {mse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

    all_metrics.append((label, mse, mae, r2))
    all_predictions[label] = (y[:200], y_pred[:200], residuals[:200])  # Limit to 200 for plotting

# Plot combined predictions vs actuals
plt.figure(figsize=(12, 6))
for label, (y_true, y_pred, _) in all_predictions.items():
    plt.plot(y_true, label=f"{label} Actual", linestyle='--')
    plt.plot(y_pred, label=f"{label} Predicted")
plt.title("Predicted vs Actual Temperature (First 200 Samples)")
plt.xlabel("Time Step")
plt.ylabel("Temperature")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "combined_predictions.png")
plt.close()

# Plot residuals
plt.figure(figsize=(12, 6))
for label, (_, _, residuals) in all_predictions.items():
    plt.plot(residuals, label=f"{label} Residuals")
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.title("Residuals (Prediction Error)")
plt.xlabel("Time Step")
plt.ylabel("Residual (Actual - Predicted)")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "residuals_plot.png")
plt.close()

# Plot error distributions
plt.figure(figsize=(12, 6))
for label, (_, _, residuals) in all_predictions.items():
    plt.hist(residuals, bins=40, alpha=0.5, label=f"{label}")
plt.title("Error Distribution Across Models")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "residuals_distribution.png")
plt.close()

# Save metrics to CSV
metrics_df = pd.DataFrame(all_metrics, columns=["Model", "MSE", "MAE", "R2"])
metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
print("Saved all plots and metrics.")
