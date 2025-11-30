import matplotlib
matplotlib.use("Agg")  # HPC-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# -----------------------------
# Translate + Noise modes
# -----------------------------
translate_modes = [
    "NT", # Not Translated
    "FT", # Fully Translated
    "PT", # Fully Translated + Prompt Translate
    "PPD", # Fully translated + Post-Process Different
    "PPS", # Fully translated + Post-Process Same
    "PTPS", # Fully Translated + Prompt Translate + Post-Process Same
    "PAR", # Partially Translated
]

noise_modes = ["NO_NOISE", "PARAPHRASE", "SYNONYM"]

# Mapping from file naming conventions to display names
translate_mode_mapping = {
    "": "NT",      # Not Translated (no postfix)
    "_f": "FT",    # Fully Translated
    "_pt": "PT",   # Prompt Translate
    "_ppd": "PPD", # Post-Process Different
    "_pps": "PPS", # Post-Process Same
    "_ptps": "PTPS", # Prompt Translate + Post-Process Same
    "_par": "PAR" # Partially Translated
}

noise_mode_mapping = {
    "": "NO_NOISE",  # No noise (no postfix)
    "_para": "PARAPHRASE",  # Paraphrase
    "_syno": "SYNONYM"      # Synonym
}


def generate_heatmap(model_name: str, output_dir: str = ".", result_dir: str = "result/score") -> None:
    """
    Generate a heatmap for a given model showing accuracy across translate and noise modes.

    Args:
        model_name: The model directory name (e.g., "gpt-5", "gpt-5-mini", "gpt-5-nano")
        output_dir: Directory to save the heatmap image (default: current directory)
        result_dir: Directory containing the score files (default: "result/score")
    """

    # Initialize data structure: dict[translate_mode][noise_mode] = accuracy
    data_dict = {}
    for tm in translate_modes:
        data_dict[tm] = {}
        for nm in noise_modes:
            data_dict[tm][nm] = None

    # Path to model's score directory
    model_score_dir = Path(result_dir) / model_name

    if not model_score_dir.exists():
        print(f"Error: Model directory '{model_score_dir}' does not exist")
        return

    # The new file naming convention from main.py is:
    # - vanilla.json (for NT + NO_NOISE)
    # - {language_postfix}{mode_postfix}{noise_postfix}.json
    # Example: zh_f.json, zh_par_para.json, etc.

    for score_file in model_score_dir.glob("*.json"):
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                data = json.loads(first_line)
                accuracy = data.get("accuracy")

                if accuracy is None:
                    print(f"Warning: No 'accuracy' field in {score_file.name}")
                    continue

                # Extract translate and noise modes from filename
                filename = score_file.stem  # Remove .json

                # Special case: vanilla.json means NT + NO_NOISE
                if filename == "vanilla":
                    translate_mode = "NT"
                    noise_mode = "NO_NOISE"
                    data_dict[translate_mode][noise_mode] = accuracy
                    print(f"Loaded {score_file.name}: {translate_mode} + {noise_mode} = {accuracy:.3f}")
                    continue

                # Parse the filename: {language_postfix}{mode_postfix}{noise_postfix}
                # Language postfix: zh_, hi_ (we ignore the language, just look for mode and noise)
                # Translate mode postfix: f, pt, ppd, pps, ptps, par
                # Noise postfix: _para, _syno

                translate_mode_str = ""
                noise_mode_str = ""

                # Remove language prefix if present (zh_, hi_)
                if filename.startswith("zh_"):
                    filename = filename[3:]  # Remove "zh_"
                elif filename.startswith("hi_"):
                    filename = filename[3:]  # Remove "hi_"

                # Extract noise mode (check for _para or _syno at the end)
                if filename.endswith("_para"):
                    noise_mode_str = "_para"
                    filename = filename[:-5]  # Remove _para
                elif filename.endswith("_syno"):
                    noise_mode_str = "_syno"
                    filename = filename[:-5]  # Remove _syno

                # What remains should be the translate mode
                # Check for translate mode postfixes
                if filename == "ptps":
                    translate_mode_str = "_ptps"
                elif filename == "ppd":
                    translate_mode_str = "_ppd"
                elif filename == "pps":
                    translate_mode_str = "_pps"
                elif filename == "pt":
                    translate_mode_str = "_pt"
                elif filename == "f":
                    translate_mode_str = "_f"
                elif filename == "par":
                    translate_mode_str = "_par"
                elif filename == "":
                    # No translate mode means NT (but we already handled vanilla above)
                    translate_mode_str = ""
                else:
                    print(f"Warning: Unknown translate mode in filename: {score_file.name}")
                    continue

                # Convert to display names
                translate_mode = translate_mode_mapping.get(translate_mode_str, "UNKNOWN")
                noise_mode = noise_mode_mapping.get(noise_mode_str, "UNKNOWN")

                if translate_mode != "UNKNOWN" and noise_mode != "UNKNOWN":
                    data_dict[translate_mode][noise_mode] = accuracy
                    print(f"Loaded {score_file.name}: {translate_mode} + {noise_mode} = {accuracy:.3f}")
                else:
                    print(f"Warning: Could not parse {score_file.name} (translate: {translate_mode}, noise: {noise_mode})")

        except Exception as e:
            print(f"Error reading {score_file.name}: {e}")

    # Convert to DataFrame
    data = []
    for tm in translate_modes:
        row = []
        for nm in noise_modes:
            value = data_dict[tm][nm]
            row.append(value if value is not None else np.nan)
        data.append(row)

    df = pd.DataFrame(data, index=translate_modes, columns=noise_modes)

    # Transpose the dataframe for heatmap visualization
    df = df.T

    # Check if we have any data
    if df.isna().all().all():
        print(f"Error: No valid data found for model '{model_name}'")
        return

    # Print summary
    print(f"\nData for model '{model_name}':")
    print(df)

    # Plot heatmap
    plt.figure(figsize=(8, 5))

    # Use a lighter, pleasant colormap
    plt.imshow(df, cmap="RdYlGn", interpolation="nearest", vmin=0.0, vmax=1.0)

    # Colorbar
    plt.colorbar(label="Accuracy")

    # Ticks (transposed: translate modes on x-axis, noise modes on y-axis)
    plt.xticks(np.arange(len(translate_modes)), translate_modes, rotation=45)
    plt.yticks(np.arange(len(noise_modes)), noise_modes)

    # Annotate values in each grid cell
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iloc[i, j]
            # Only annotate if we have data
            if not pd.isna(value):
                plt.text(
                    j, i,
                    f"{value:.3f}",             # round to 3 decimals
                    ha="center", va="center",
                    color="black", fontsize=9   # black text = readable on light colormap
                )

    plt.title(f"Heatmap: {model_name} - Translate Mode × Noise Mode")
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"heatmap_{model_name}.png")
    plt.savefig(output_path)
    print(f"\nSaved heatmap to {output_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    # Generate heatmaps for different models
    # Model names should match the directory names in result/score/
    # Examples: "gpt-5", "gpt-5-mini", "gpt-5-nano"
    models = ["gpt-5"]

    for model in models:
        print(f"\n{'='*60}")
        print(f"Generating heatmap for {model}")
        print(f"{'='*60}")
        generate_heatmap(model)