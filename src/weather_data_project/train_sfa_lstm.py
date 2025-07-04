import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Multiply, Softmax, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.activations import tanh

# Configuration
sequence_length = 24
target_column = "temperature"
test_size = 0.1
random_seed = 42

# Paths
project_root = Path(__file__).resolve().parents[2]
processed_dir = project_root / "data" / "processed"
output_dir = project_root / "models"
output_dir.mkdir(parents=True, exist_ok=True)

# Spatial Feature Attention
def spatial_attention_block(inputs):
    # inputs: (batch_size, timesteps, features)
    attention = Dense(inputs.shape[-1], activation='tanh')(inputs)  # (batch_size, timesteps, features)
    attention = Softmax(axis=-1)(attention)                         # same shape, softmax over features
    attended = Multiply()([inputs, attention])                      # (batch_size, timesteps, features)
    return attended

# Build SFA-LSTM model
def build_sfa_lstm(input_shape):
    inputs = Input(shape=input_shape)
    attention_out = spatial_attention_block(inputs)
    lstm_out = LSTM(32)(attention_out)
    output = Dense(1, activation='linear')(lstm_out)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Load dataset
def load_dataset(csv_file, has_city=False):
    df = pd.read_csv(csv_file)
    if has_city:
        df = df.drop(columns=['city'])
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    n_features = int(X.shape[1] / sequence_length)
    X = X.reshape((X.shape[0], sequence_length, n_features))
    return X, y

# Train on separate cities
for file in processed_dir.glob("*_weather_2013_2023.csv"):
    print(f"Training on: {file.name}")
    X, y = load_dataset(file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    model = build_sfa_lstm(X.shape[1:])
    model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=0)

    y_pred = model.predict(X_test)
    print(f"Results for {file.stem}")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))
    print()

# Train on combined dataset
combined_file = processed_dir / "combined_cities_weather_2013_2023.csv"
if combined_file.exists():
    print("Training on combined dataset")
    X, y = load_dataset(combined_file, has_city=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    model = build_sfa_lstm(X.shape[1:])
    model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=0)

    y_pred = model.predict(X_test)
    print("Results for combined dataset")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))
