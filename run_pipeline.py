import argparse
import subprocess
import sys
import time
from pathlib import Path

def run(script_name, label):
    print(f"\n{'='*40}\nStep: {label}\n{'='*40}")
    start = time.time()
    try:
        subprocess.run(
            ["poetry", "run", "python", str(Path("src/weather_data_project") / script_name)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during step '{label}': {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Completed '{label}' in {time.time() - start:.1f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the weather data pipeline.")
    parser.add_argument("--only", nargs="+", help="Run only these steps (e.g. download clean train)")
    parser.add_argument("--skip", nargs="+", help="Skip these steps (e.g. download train)")
    args = parser.parse_args()

    steps = [
        ("download_weather_openmeteo.py", "Downloading weather data", "download"),
        ("clean_data.py", "Cleaning raw data", "clean"),
        ("prepare_sequences.py", "Preparing sequences", "prepare"),
        ("prepare_combined_sequences.py", "Combining datasets", "combine"),
        ("train_sfa_lstm.py", "Training SFA-LSTM models", "train"),
        ("visualize_predictions.py", "Visualizing predictions", "visualize"),
        ("advanced_visualize_predictions.py", "Visualizing advanced predictions", "advanced-visualize"),
        # ("visualize_attention_weights.py", "Visualizing attention weights", "attention"),
    ]

    if args.only:
        only_set = set(args.only)
        steps_to_run = [s for s in steps if s[2] in only_set]
    else:
        skip_set = set(args.skip) if args.skip else set()
        steps_to_run = [s for s in steps if s[2] not in skip_set]

    for script, label, _ in steps_to_run:
        run(script, label)

    print("\nPipeline complete.")
