"""
LLM as Judge: Exploring the relationship between perplexity and preference
"""
import os

from judge.collect_perplexity_local import collect_perplexity_local
from judge.collect_preference_local_direct import collect_preference_local_direct
from judge.collect_preference_local_cot import collect_preference_local_cot
from judge.generate_dataset import generate_answer_datasets
from util import get_model_directory_name
os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"
import sys
import importlib.util
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import json
import re
import math
import argparse
from judge.parse_dataset import parse_dataset, prepare_answer_pairs_bilingual

from config import LocalModel, ResultType
from models import create_backend, create_interface

# Set UTF-8 encoding for console output (Windows fix)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

backend_type = "vllm"  # or "huggingface"

def get_or_create_backend(model_name, device, backend_type: str, num_gpus: int):
    """
    Get or create a cached model backend using the new unified factory.

    The backend is cached automatically by the factory and reused if possible.

    Args:
        model_name: HuggingFace model name
        device: Device to use ("cuda" or "cpu")
        backend_type: Backend type ("huggingface" or "vllm")
        num_gpus: Number of GPUs to use

    Returns:
        Cached or newly created ModelBackend instance
    """
    backend = create_backend(
        backend_type=backend_type,
        model_name=model_name,
        device=device,
        num_gpus=num_gpus
    )
    return backend


def find_assistant_answer_start(full_ids, prefix_ids):
    """
    Find the start position of assistant answer in the token sequence.

    Args:
        full_ids: List of all token IDs in the full conversation
        prefix_ids: List of token IDs representing the assistant prefix

    Returns:
        Index of the first token after the prefix

    Raises:
        ValueError: If prefix not found in full_ids
    """
    L = len(prefix_ids)
    for i in range(len(full_ids) - L + 1):
        if full_ids[i:i+L] == prefix_ids:
            return i + L  # the first token *after* the prefix
    raise ValueError("Assistant prefix not found in full_ids.")


def load_entries(file_path):
    """Load entries from a JSONL file."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def combine_entries_to_pairs(entries1, entries2, lang1, lang2):
    """
    Combine two lists of entries into pairs by matching indices.

    Args:
        entries1: List of entries for answer1 (lang1)
        entries2: List of entries for answer2 (lang2)
        lang1: Language code for answer1
        lang2: Language code for answer2

    Returns:
        List of pairs with structure:
        {
            'index': int,
            'question': str,
            'answer1': str,
            'answer2': str,
            'lang1': str,
            'lang2': str,
            'is_correct1': bool,
            'is_correct2': bool,
            'subject': str,
        }
    """
    # Index entries by their index field
    entries1_by_index = {e['index']: e for e in entries1}
    entries2_by_index = {e['index']: e for e in entries2}

    # Find common indices
    common_indices = set(entries1_by_index.keys()) & set(entries2_by_index.keys())

    pairs = []
    for idx in sorted(common_indices):
        e1 = entries1_by_index[idx]
        e2 = entries2_by_index[idx]

        pair = {
            'index': idx,
            'question': e1['question'],
            'answer1': e1['answer'],
            'answer2': e2['answer'],
            'lang1': lang1,
            'lang2': lang2,
            'is_correct1': e1['is_correct'],
            'is_correct2': e2['is_correct'],
            'subject': e1.get('subject', ''),
        }
        pairs.append(pair)

    return pairs


def compare_results(preferences, perplexities_lang1, perplexities_lang2, lang1, lang2):
    """
    Compare preferences with perplexity rankings and calculate correlation.
    """
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)

    # Determine which answer has lower perplexity (better)
    perplexity_preferences = []
    for p1, p2 in zip(perplexities_lang1, perplexities_lang2):
        if p1 is not None and p2 is not None:
            perplexity_preferences.append(1 if p1 < p2 else 2)
        else:
            perplexity_preferences.append(None)

    # Calculate agreement
    valid_comparisons = 0
    agreement = 0
    lang1_preference_count = 0
    lang2_preference_count = 0
    lang1_perplexity_count = 0
    lang2_perplexity_count = 0

    for pref, perp_pref in zip(preferences, perplexity_preferences):
        if pref is not None and perp_pref is not None:
            valid_comparisons += 1

            if pref == 1:
                lang1_preference_count += 1
            else:
                lang2_preference_count += 1

            if perp_pref == 1:
                lang1_perplexity_count += 1
            else:
                lang2_perplexity_count += 1

            if pref == perp_pref:
                agreement += 1

    print(f"\nTotal valid comparisons: {valid_comparisons}")
    print(f"\nPreference Method (which answer LLM judges as better):")
    print(f"  {lang1} answers preferred: {lang1_preference_count}/{valid_comparisons} = {100*lang1_preference_count/valid_comparisons:.2f}%")
    print(f"  {lang2} answers preferred: {lang2_preference_count}/{valid_comparisons} = {100*lang2_preference_count/valid_comparisons:.2f}%")

    print(f"\nPerplexity Method (which answer has lower perplexity):")
    print(f"  {lang1} answers have lower perplexity: {lang1_perplexity_count}/{valid_comparisons} = {100*lang1_perplexity_count/valid_comparisons:.2f}%")
    print(f"  {lang2} answers have lower perplexity: {lang2_perplexity_count}/{valid_comparisons} = {100*lang2_perplexity_count/valid_comparisons:.2f}%")

    print(f"\nAgreement between methods:")
    print(f"  {agreement}/{valid_comparisons} = {100*agreement/valid_comparisons:.2f}%")

    # Calculate correlation
    from scipy.stats import pearsonr, spearmanr

    # Filter out None values
    valid_indices = [i for i in range(len(preferences))
                     if preferences[i] is not None and perplexity_preferences[i] is not None]

    if len(valid_indices) > 1:
        pref_values = [preferences[i] for i in valid_indices]
        perp_values = [perplexity_preferences[i] for i in valid_indices]

        try:
            # Check if there's variance
            if len(set(pref_values)) > 1 and len(set(perp_values)) > 1:
                pearson_corr, pearson_p = pearsonr(pref_values, perp_values)
                spearman_corr, spearman_p = spearmanr(pref_values, perp_values)

                print(f"\nCorrelation statistics:")
                print(f"  Pearson correlation: {pearson_corr:.3f} (p={pearson_p:.4f})")
                print(f"  Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.4f})")
            else:
                print("\nCould not calculate correlation (insufficient variance)")
        except Exception as e:
            print(f"\nCould not calculate correlation: {e}")

    print("\n" + "="*60)

    # Show some examples
    print("\nExample comparisons (first 5):")
    for i in range(min(5, len(preferences))):
        if preferences[i] is not None and perplexity_preferences[i] is not None:
            print(f"\nSample {i}:")
            print(f"  Preference method chose: {lang1 if preferences[i] == 1 else lang2}")
            print(f"  Perplexity method chose: {lang1 if perplexity_preferences[i] == 1 else lang2}")
            print(f"  Match: {'✓' if preferences[i] == perplexity_preferences[i] else '✗'}")


def load_configs_from_file(config_file_path: str):
    """
    Load the 'configs' list from a specified Python file.

    Args:
        config_file_path: Path to the Python file containing configs

    Returns:
        The configs list from the specified file
    """
    # Convert to absolute path if relative
    config_file_path = os.path.abspath(config_file_path)

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("custom_configs", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if not hasattr(config_module, 'configs'):
        raise AttributeError(f"Config file {config_file_path} does not contain a 'configs' variable")

    return config_module.configs


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLM as Judge: Exploring the relationship between perplexity and preference')
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a Python file containing the 'configs' list (default: use configs from config.py)"
    )
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use for model inference (default: 1)')
    args = parser.parse_args()

    # Load configs from specified file or use default from config.py
    if args.config:
        print(f"Loading configs from: {args.config}")
        configs = load_configs_from_file(args.config)
    else:
        # configs is already imported from config.py
        print("Error: Please specify a config file using --config argument. For example, --config judge_config1.py")
        exit(1)

    print(f"Using {args.num_gpus} GPU(s)")

    # Create result directory if it doesn't exist
    os.makedirs("judge/result", exist_ok=True)

    for config in configs:
        print("Processing configuration: ", config)

        # Determine alphabetical order for language codes
        sorted_langs = sorted([config.lang1, config.lang2])
        first_lang = sorted_langs[0]
        second_lang = sorted_langs[1]

        # Load individual entry datasets
        entries_lang1_correct = load_entries(f"judge/datasets/{first_lang}_correct.jsonl")
        entries_lang1_incorrect = load_entries(f"judge/datasets/{first_lang}_incorrect.jsonl")
        entries_lang2_correct = load_entries(f"judge/datasets/{second_lang}_correct.jsonl")
        entries_lang2_incorrect = load_entries(f"judge/datasets/{second_lang}_incorrect.jsonl")

        print(f"Loaded {len(entries_lang1_correct)} entries from judge/datasets/{first_lang}_correct.jsonl")
        print(f"Loaded {len(entries_lang1_incorrect)} entries from judge/datasets/{first_lang}_incorrect.jsonl")
        print(f"Loaded {len(entries_lang2_correct)} entries from judge/datasets/{second_lang}_correct.jsonl")
        print(f"Loaded {len(entries_lang2_incorrect)} entries from judge/datasets/{second_lang}_incorrect.jsonl")

        # Combine entries into pairs for preference methods
        # 4 pair combinations:
        # 1. lang1 correct, lang2 incorrect
        # 2. lang1 incorrect, lang2 correct
        # 3. both correct
        # 4. both incorrect
        pairs_lang1_correct_lang2_incorrect = combine_entries_to_pairs(
            entries_lang1_correct, entries_lang2_incorrect, first_lang, second_lang
        )
        pairs_lang1_incorrect_lang2_correct = combine_entries_to_pairs(
            entries_lang1_incorrect, entries_lang2_correct, first_lang, second_lang
        )
        pairs_both_correct = combine_entries_to_pairs(
            entries_lang1_correct, entries_lang2_correct, first_lang, second_lang
        )
        pairs_both_incorrect = combine_entries_to_pairs(
            entries_lang1_incorrect, entries_lang2_incorrect, first_lang, second_lang
        )

        print(f"Created {len(pairs_lang1_correct_lang2_incorrect)} pairs for {first_lang}_correct_{second_lang}_incorrect")
        print(f"Created {len(pairs_lang1_incorrect_lang2_correct)} pairs for {first_lang}_incorrect_{second_lang}_correct")
        print(f"Created {len(pairs_both_correct)} pairs for {first_lang}_correct_{second_lang}_correct")
        print(f"Created {len(pairs_both_incorrect)} pairs for {first_lang}_incorrect_{second_lang}_incorrect")

        # Get or create backend with caching
        model_name = config.model.value
        display_model_name = get_model_directory_name(config.model)

        # Get or create backend (batch size will be calculated automatically)
        backend = get_or_create_backend(
            model_name=model_name,
            device="cuda",
            backend_type=backend_type,
            num_gpus=args.num_gpus
        )

        # Create model interface for model-specific behavior
        model_interface = create_interface(model_name)
        print(f"Using model interface: {model_interface.__class__.__name__}")

        match config.result_type:
            case ResultType.PREFERENCE_DIRECT:
                # Process pairs for preference_direct
                for pairs, dataset_suffix in [
                    (pairs_lang1_correct_lang2_incorrect, f"{first_lang}_correct_{second_lang}_incorrect"),
                    (pairs_lang1_incorrect_lang2_correct, f"{first_lang}_incorrect_{second_lang}_correct"),
                    (pairs_both_correct, f"{first_lang}_correct_{second_lang}_correct"),
                    (pairs_both_incorrect, f"{first_lang}_incorrect_{second_lang}_incorrect")
                ]:
                    output_dir = f"judge/result/{display_model_name}/preferences_local_direct"
                    os.makedirs(output_dir, exist_ok=True)
                    collect_preference_local_direct(
                        pairs=pairs,
                        backend=backend,
                        model_interface=model_interface,
                        output_file=f"{output_dir}/{dataset_suffix}.jsonl"
                    )

            case ResultType.PREFERENCE_COT:
                # Process pairs for preference_cot
                for pairs, dataset_suffix in [
                    (pairs_lang1_correct_lang2_incorrect, f"{first_lang}_correct_{second_lang}_incorrect"),
                    (pairs_lang1_incorrect_lang2_correct, f"{first_lang}_incorrect_{second_lang}_correct"),
                    (pairs_both_correct, f"{first_lang}_correct_{second_lang}_correct"),
                    (pairs_both_incorrect, f"{first_lang}_incorrect_{second_lang}_incorrect")
                ]:
                    output_dir = f"judge/result/{display_model_name}/preferences_local_cot"
                    os.makedirs(output_dir, exist_ok=True)
                    collect_preference_local_cot(
                        pairs=pairs,
                        backend=backend,
                        model_interface=model_interface,
                        output_file=f"{output_dir}/{dataset_suffix}.jsonl",
                        batch_size=1
                    )

            case ResultType.PERPLEXITY:
                # Process individual entries for perplexity
                for entries, entry_suffix in [
                    (entries_lang1_correct, f"{first_lang}_correct"),
                    (entries_lang1_incorrect, f"{first_lang}_incorrect"),
                    (entries_lang2_correct, f"{second_lang}_correct"),
                    (entries_lang2_incorrect, f"{second_lang}_incorrect")
                ]:
                    output_dir = f"judge/result/{display_model_name}/perplexities_local"
                    os.makedirs(output_dir, exist_ok=True)
                    collect_perplexity_local(
                        entries=entries,
                        backend=backend,
                        model_interface=model_interface,
                        output_file=f"{output_dir}/{entry_suffix}.jsonl"
                    )
            case _:
                print(f"Unknown result type: {config.result_type}")
                raise ValueError(f"Unknown result type: {config.result_type}")
        print("Collected results for configuration: ", config)
