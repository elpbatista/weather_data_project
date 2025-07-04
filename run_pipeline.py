import argparse
import subprocess

def run(script_name, label):
    print(f"Step: {label}")
    subprocess.run(["poetry", "run", "python", f"src/weather_data_project/{script_name}"], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the weather data pipeline.")
    parser.add_argument("--skip-download", action="store_true", help="Skip weather data download")
    parser.add_argument("--skip-clean", action="store_true", help="Skip data cleaning")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip sequence preparation")
    parser.add_argument("--skip-combine", action="store_true", help="Skip dataset combining")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-visualize", action="store_true", help="Skip prediction visualization")
    # parser.add_argument("--skip-attention", action="store_true", help="Skip attention visualization")
    args = parser.parse_args()

    if not args.skip_download:
        run("download_weather_openmeteo.py", "Downloading weather data")

    if not args.skip_clean:
        run("clean_data.py", "Cleaning raw data")

    if not args.skip_prepare:
        run("prepare_sequences.py", "Preparing sequences")

    if not args.skip_combine:
        run("combine_cities_sequences.py", "Combining datasets")

    if not args.skip_train:
        run("train_sfa_lstm.py", "Training SFA-LSTM models")

    if not args.skip_visualize:
        run("advanced_visualize_predictions.py", "Visualizing predictions")

    # if not args.skip_attention:
    #     run("visualize_attention_weights.py", "Visualizing attention weights")

    print("Pipeline complete.")
