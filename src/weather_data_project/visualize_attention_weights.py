import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import Model # type: ignore

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

def extract_attention_model(full_model):
    """Return a model that outputs attention weights."""
    attention_layer_output = full_model.layers[2].output  # Dense softmax output in spatial_attention_block
    attention_model = Model(inputs=full_model.input, outputs=attention_layer_output)
    return attention_model

def visualize_attention_weights(model_path, data_path, has_city, label):
    model = load_model(model_path)
    attention_model = extract_attention_model(model)

    X, _ = load_dataset(data_path, has_city=has_city)
    attn_weights = attention_model.predict(X[:1])  # Use the first sample only
    attn_weights = attn_weights[0]  # shape: (timesteps, features)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(attn_weights.T, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    plt.xlabel("Time Step")
    plt.ylabel("Feature Index")
    plt.title(f"Spatial Attention Weights - {label}")
    plt.tight_layout()
    plt.savefig(output_dir / f"attention_weights_{label.lower()}.png")
    plt.close()

# Process each model
for model_file in models_dir.glob("*.keras"):
    model_name = model_file.stem
    print(f"Processing attention weights for: {model_file.name}")

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

    visualize_attention_weights(model_file, data_file, has_city, label)
    print(f"Saved attention visualization for {label}")
