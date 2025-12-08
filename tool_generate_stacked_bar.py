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
    "PAR", # Partially Translated
    "FT", # Fully Translated
    "PT", # Fully Translated + Prompt Translate
    "PRE", # Fully Translated + Pre-Translate
    "POST", # Fully Translated + Post-Translate
]

noise_modes = ["NO_NOISE", "PARAPHRASE", "SYNONYM"]

# Error categories (from ToolErrorCategory enum)
error_categories = [
    "syntax_error",
    "misc_errors",
    "wrong_values",
    "language_mismatch",
    "relevant_but_incorrect",
    "exactly_same_meaning",
    "other_errors",
]

# Color map for each error category
category_colors = {
    "syntax_error": "#e74c3c",  # Red
    "misc_errors": "#e67e22",  # Orange
    "wrong_values": "#f39c12",  # Yellow-orange
    "language_mismatch": "#9b59b6",  # Purple
    "relevant_but_incorrect": "#3498db",  # Blue
    "exactly_same_meaning": "#2ecc71",  # Green
    "other_errors": "#95a5a6",  # Gray
}


def generate_stacked_bar_chart(model_name: str, output_dir: str, result_dir: str,
                                selected_translate_mode: str = None,
                                selected_noise_mode: str = None) -> None:
    """
    Generate a stacked bar chart for a given model showing error type distributions.

    Args:
        model_name: The model directory name (e.g., "gpt-5", "gpt-5-mini", "gpt-5-nano")
        output_dir: Directory to save the chart image (default: current directory)
        result_dir: Directory containing the categorize_score files (default: "tool/result/categorize_score")
        selected_translate_mode: If specified, only show bars for this translate mode across noise modes
        selected_noise_mode: If specified, only show bars for this noise mode across translate modes
    """

    # Initialize data structure: dict[translate_mode][noise_mode][category] = count
    data_dict = {}
    for tm in translate_modes:
        data_dict[tm] = {}
        for nm in noise_modes:
            data_dict[tm][nm] = {cat: 0 for cat in error_categories}

    # Path to model's categorize_score directory
    model_categorize_dir = Path(result_dir) / model_name

    if not model_categorize_dir.exists():
        print(f"Error: Model directory '{model_categorize_dir}' does not exist")
        return

    # Parse categorize_score files
    for score_file in model_categorize_dir.glob("*.json"):
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                summary = data.get("summary")

                if summary is None:
                    print(f"Warning: No 'summary' field in {score_file.name}")
                    continue

                # Extract translate and noise modes from filename
                filename = score_file.stem  # Remove .json extension

                # Parse the filename by splitting on underscore
                if filename.startswith("_"):
                    filename = filename[1:]

                tags = filename.split("_")

                if len(tags) != 6:
                    print(f"Warning: Unexpected filename format '{score_file.name}' (expected 6 tags, got {len(tags)})")
                    continue

                language_tag = tags[0]
                translate_level_tag = tags[1]
                pre_translate_tag = tags[2]
                noise_tag = tags[3]
                prompt_translate_tag = tags[4]
                post_translate_tag = tags[5]

                # Map noise_tag to noise_mode
                if noise_tag == "nonoise":
                    noise_mode = "NO_NOISE"
                elif noise_tag == "para":
                    noise_mode = "PARAPHRASE"
                elif noise_tag == "syno":
                    noise_mode = "SYNONYM"
                else:
                    print(f"Warning: Unknown noise tag '{noise_tag}' in {score_file.name}")
                    continue

                # Map combination of tags to translate_mode (same logic as heatmap)
                if language_tag == "en" and translate_level_tag == "na":
                    translate_mode = "NT"
                elif language_tag in ["zh", "hi"] and translate_level_tag == "parttrans":
                    translate_mode = "PAR"
                elif language_tag in ["zh", "hi"] and translate_level_tag == "fulltrans":
                    if (pre_translate_tag == "nopretrans" and prompt_translate_tag == "noprompt" and
                        post_translate_tag == "noposttrans"):
                        translate_mode = "FT"
                    elif (pre_translate_tag == "nopretrans" and prompt_translate_tag == "prompt" and
                          post_translate_tag == "noposttrans"):
                        translate_mode = "PT"
                    elif (pre_translate_tag == "pretrans" and prompt_translate_tag == "noprompt" and
                          post_translate_tag == "noposttrans"):
                        translate_mode = "PRE"
                    elif (pre_translate_tag == "nopretrans" and prompt_translate_tag == "noprompt" and
                          post_translate_tag == "posttrans"):
                        translate_mode = "POST"
                    else:
                        print(f"Warning: Unknown tag combination in {score_file.name}")
                        continue
                else:
                    print(f"Error: Unknown language/translate_level combination in {score_file.name}")
                    exit(1)

                # Store the category counts
                for category, count in summary.items():
                    if category in error_categories:
                        data_dict[translate_mode][noise_mode][category] = count

                print(f"Loaded {score_file.name}: {translate_mode} + {noise_mode}")

        except Exception as e:
            print(f"Error reading {score_file.name}: {e}")

    # Prepare data for plotting
    if selected_translate_mode and selected_noise_mode:
        print("Error: Cannot specify both translate_mode and noise_mode")
        return
    elif selected_translate_mode:
        # Show bars for selected translate mode across noise modes
        bar_labels = noise_modes
        bar_data = []
        for nm in noise_modes:
            category_counts = data_dict[selected_translate_mode][nm]
            bar_data.append(category_counts)
        title = f"{model_name} - {selected_translate_mode} across Noise Modes"
        output_name = f"stacked_bar_{model_name}_{selected_translate_mode}.png"
    elif selected_noise_mode:
        # Show bars for selected noise mode across translate modes
        bar_labels = translate_modes
        bar_data = []
        for tm in translate_modes:
            category_counts = data_dict[tm][selected_noise_mode]
            bar_data.append(category_counts)
        title = f"{model_name} - {selected_noise_mode} across Translate Modes"
        output_name = f"stacked_bar_{model_name}_{selected_noise_mode}.png"
    else:
        # Default: show all combinations (might be too many bars)
        bar_labels = []
        bar_data = []
        for tm in translate_modes:
            for nm in noise_modes:
                bar_labels.append(f"{tm}_{nm}")
                category_counts = data_dict[tm][nm]
                bar_data.append(category_counts)
        title = f"{model_name} - All Combinations"
        output_name = f"stacked_bar_{model_name}_all.png"

    # Create DataFrame for easier plotting
    df_data = []
    for counts in bar_data:
        df_data.append([counts[cat] for cat in error_categories])

    df = pd.DataFrame(df_data, index=bar_labels, columns=error_categories)

    # Check if we have any data
    if df.sum().sum() == 0:
        print(f"Error: No error data found for model '{model_name}'")
        return

    # Print summary
    print(f"\nError distribution for model '{model_name}':")
    print(df)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert counts to rates by dividing by 200
    df_rate = df / 200.0

    # Create stacked bars
    bottom = np.zeros(len(bar_labels))
    for category in error_categories:
        values = df_rate[category].values
        ax.bar(bar_labels, values, label=category, bottom=bottom,
               color=category_colors[category], edgecolor='white', linewidth=0.5)
        bottom += values

    # Calculate totals for each bar (as rates)
    totals = df_rate.sum(axis=1).values

    # Add total numbers on top of each bar
    for i, (label, total) in enumerate(zip(bar_labels, totals)):
        if total > 0:  # Only annotate if there's data
            ax.text(i, total, f'{total:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Set y-axis range to be slightly higher than the largest bar
    max_total = totals.max()
    if max_total > 0:
        ax.set_ylim(0, max_total * 1.1)  # 10% higher than max

    # Customize plot
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)

    # Rotate x-axis labels if there are many bars
    if len(bar_labels) > 10:
        plt.xticks(rotation=45, ha='right')
    else:
        plt.xticks(rotation=0)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved stacked bar chart to {output_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    # Generate stacked bar charts for different models
    models = ["gpt-5-nano"]  # Add more models as needed

    for model in models:
        print(f"\n{'='*60}")
        print(f"Generating stacked bar charts for {model}")
        print(f"{'='*60}")

        # Generate chart for NO_NOISE across all translate modes
        generate_stacked_bar_chart(
            model,
            ".",
            "tool/result/categorize_score",
            selected_noise_mode="NO_NOISE"
        )

        # Generate chart for PARAPHRASE across all translate modes
        generate_stacked_bar_chart(
            model,
            ".",
            "tool/result/categorize_score",
            selected_noise_mode="PARAPHRASE"
        )

        # Generate chart for SYNONYM across all translate modes
        generate_stacked_bar_chart(
            model,
            ".",
            "tool/result/categorize_score",
            selected_noise_mode="SYNONYM"
        )

        # Or generate chart for a specific translate mode across noise modes
        # generate_stacked_bar_chart(
        #     model,
        #     ".",
        #     "tool/result/categorize_score",
        #     selected_translate_mode="NT"
        # )
