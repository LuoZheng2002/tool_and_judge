from tool.parse_dataset import load_json_lines
import json
import os
import sys
import argparse
import asyncio
import importlib.util
from config import *
from tool.parse_ast import *
import re
from models import create_backend, create_interface
from tool_categorize import categorize_single_sample_async
from models.name_mapping import get_global_name_mapper
from filelock import FileLock

from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# LOCAL MODEL INFERENCE BACKEND CONFIGURATION
# ============================================================================
# Set to True to use vLLM backend for local models (faster, better batching)
# Set to False to use HuggingFace transformers backend (default)
USE_VLLM_BACKEND = True
# ============================================================================


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


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run BFCL evaluation with custom configuration"
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to a Python file containing the 'configs' list (default: use configs from config.py)"
)
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for local inference (default: 1)"
)
args = parser.parse_args()

# Load configs from specified file or use default from config.py
if args.config:
    print(f"Loading configs from: {args.config}")
    configs = load_configs_from_file(args.config)
else:
    # configs is already imported from config.py via 'from config import *'
    # print("Using configs from config.py")
    print("Error: Please specify a config file using --config argument. For example, --config config1.py")
    exit(1)

# File operation helper functions

def load_json_lines_with_id(file_path: str) -> tuple[list, set]:
    """
    Load JSON lines from a file and extract existing IDs.

    Args:
        file_path: Path to the JSON lines file

    Returns:
        Tuple of (results_list, existing_ids_set)
    """
    results = []
    existing_ids = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        if f.readable():
            for line in f:
                line_json = json.loads(line)
                id = line_json["id"]
                results.append(line_json)
                existing_ids.add(id)
    return results, existing_ids

def load_json_lines(file_path: str) -> list:
    """
    Load JSON lines from a file and extract existing IDs.

    Args:
        file_path: Path to the JSON lines file

    Returns:
        Tuple of (results_list, existing_ids_set)
    """
    results = []

    with open(file_path, 'r', encoding='utf-8') as f:
        if f.readable():
            for line in f:
                line_json = json.loads(line)
                results.append(line_json)
    return results


def write_json_lines_to_file(file_path: str, results: list) -> None:
    """
    Write JSON lines to a file, overwriting existing content.
    Creates parent directory if it doesn't exist.

    Args:
        file_path: Path to the output file
        results: List of dictionaries to write
    """
    # Create parent directory if it doesn't exist
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()


def sort_results_by_id(results: list) -> list:
    """
    Sort results by numeric ID extracted from the 'id' field.

    Args:
        results: List of dictionaries with 'id' field

    Returns:
        Sorted list of results
    """
    return sorted(
        results,
        key=lambda x: int(re.search(r'\d+', x["id"]).group())
                      if re.search(r'\d+', x["id"])
                      else float('inf')
    )


def append_and_rewrite_json_lines(file_path: str, results: list) -> None:
    """
    Append results to file and rewrite entire file with sorted results.

    Args:
        file_path: Path to the output file
        results: List of dictionaries to write
    """
    sorted_results = sort_results_by_id(results)
    write_json_lines_to_file(file_path, sorted_results)


def get_model_directory_name(model: Model) -> str:
    """
    Get a filesystem-safe directory name from model enum value.

    Args:
        model: ApiModel or LocalModel enum

    Returns:
        Filesystem-safe directory name based on model's official name
    """
    model_value = model.value
    # Replace filesystem-unsafe characters
    # "/" -> "-" (for models like "Qwen/Qwen2.5-7B-Instruct")
    # ":" -> "-" (for models like "meta.llama3-1-8b-instruct-v1:0")
    safe_name = model_value.replace("/", "-").replace(":", "-")
    return safe_name


def extract_model_size_in_billions(local_model: LocalModel) -> int:
    """
    Extract model size in billions from LocalModel enum value.

    Args:
        local_model: LocalModel enum

    Returns:
        Model size in billions (e.g., 8 for 8B model)

    Raises:
        ValueError: If model size cannot be extracted from model name
    """
    model_value = local_model.value

    # Extract number followed by 'B' (case insensitive)
    # Matches patterns like "8B", "14B", "32B", "80B"
    match = re.search(r'(\d+)B', model_value, re.IGNORECASE)

    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot extract model size from model name: {model_value}")


def calculate_batch_size_for_local_model(local_model: LocalModel, num_gpus: int) -> int:
    """
    Calculate batch size for local inference based on model size and number of GPUs.

    Formula: batch_size * x = 120 * num_gpus
    Where x is the model size in billions (e.g., 8 for an 8B model)

    Args:
        local_model: LocalModel enum
        num_gpus: Number of GPUs available

    Returns:
        Calculated batch size (rounded down to nearest integer)
    """
    model_size_b = extract_model_size_in_billions(local_model)
    batch_size = (120 * num_gpus) // model_size_b

    # Ensure batch size is at least 1
    batch_size = max(1, batch_size)

    return batch_size

def get_or_create_backend(model: Model, num_gpus: int = 1, max_model_len: int = 2000, instance_name: str = "default"):
    """
    Get or create a backend for the given model.

    Args:
        model: ApiModel or LocalModel enum value
        num_gpus: Number of GPUs to use
        max_model_len: Maximum model length
        instance_name: Name for this backend instance (default: "default")
                      Use "experiment" for experiment runs

    Returns:
        The backend for the given model
    """
    model_name = model.value
    if isinstance(model, ApiModel):
        backend_type = "api"
    elif isinstance(model, LocalModel):
        backend_type = "vllm" if USE_VLLM_BACKEND else "huggingface"
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    return create_backend(
        backend_type=backend_type,
        model_name=model_name,
        device="cuda",
        num_gpus=num_gpus,
        max_model_len=max_model_len,
        instance_name=instance_name
    )


def get_or_create_model_interface(model: Model):
    """
    Get or create a model interface for the given model using the new unified factory.

    Args:
        model: ApiModel or LocalModel enum value

    Returns:
        The model interface for the given model
    """
    # The new factory handles all interface creation logic
    model_identifier = model.value if hasattr(model, 'value') else model
    return create_interface(model_identifier)


def load_category_cache(cache_path: str) -> dict:
    """Load category cache from file. Assumes lock is already acquired."""
    category_cache = {}
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Each line is a [key, value] pair
                item = json.loads(line)
                key_tuple = (item[0][0], tuple(item[0][1]))
                category_cache[key_tuple] = item[1]
    except FileNotFoundError:
        print(f"Category cache file not found. Creating new cache.")
    except json.JSONDecodeError as e:
        print(f"Error decoding category cache: {e}. Creating new cache.")
    return category_cache


def save_category_cache(cache_path: str, category_cache: dict) -> None:
    """Save category cache to file. Assumes lock is already acquired."""
    with open(cache_path, 'w', encoding='utf-8') as f:
        for key, value in category_cache.items():
            cache_entry = [[key[0], list(key[1])], value]
            f.write(json.dumps(cache_entry, ensure_ascii=False) + '\n')


async def process_all_configs():
    """Process all configs within a single event loop to allow backend reuse."""
    # Setup category cache with file locking
    category_cache_path = "tool_category_cache.json"
    category_cache_lock_path = "tool_category_cache.json.lock"
    cache_lock = FileLock(category_cache_lock_path, timeout=60)

    for config in configs:
        print(f"Processing config: {config}", flush=True)

        # map translate_info to language_postfix, translate_dataset_prefix, translate_mode_prefix
        match config.translate_mode:
            case Translated(language, option):
                match language:
                    case Language.CHINESE:
                        language_tag = "_zh"
                    case Language.HINDI:
                        language_tag = "_hi"
                match option:
                    case TranslateOption.FULLY_TRANSLATED:
                        translate_level_tag = "_fulltrans"
                        pre_translate_tag = "_nopretrans"
                        prompt_translate_tag = "_noprompt"
                        post_translate_tag = "_noposttrans"
                    case TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE:
                        translate_level_tag = "_fulltrans"
                        pre_translate_tag = "_nopretrans"
                        prompt_translate_tag = "_prompt"
                        post_translate_tag = "_noposttrans"
                    case TranslateOption.FULLY_TRANSLATED_PRE_TRANSLATE:
                        translate_level_tag = "_fulltrans"
                        pre_translate_tag = "_pretrans"
                        prompt_translate_tag = "_noprompt"
                        post_translate_tag = "_noposttrans"
                    case TranslateOption.FULLY_TRANSLATED_POST_TRANSLATE:
                        translate_level_tag = "_fulltrans"
                        pre_translate_tag = "_nopretrans"
                        prompt_translate_tag = "_noprompt"
                        post_translate_tag = "_posttrans"
                    case TranslateOption.PARTIALLY_TRANSLATED:
                        translate_level_tag = "_parttrans"
                        pre_translate_tag = "_nopretrans"
                        prompt_translate_tag = "_noprompt"
                        post_translate_tag = "_noposttrans"
                    case _:
                        raise ValueError(f"Unsupported translate option: {option}")
            case NotTranslated():
                language_tag = "_en"
                translate_level_tag = "_na"
                pre_translate_tag = "_nopretrans"
                prompt_translate_tag = "_noprompt"
                post_translate_tag = "_noposttrans"
        match config.add_noise_mode:
            case AddNoiseMode.NO_NOISE:
                noise_tag = "_nonoise" # no noise
            case AddNoiseMode.SYNONYM:
                noise_tag = "_syno" # synonym
            case AddNoiseMode.PARAPHRASE:
                noise_tag = "_para" # paraphrase
            case _:
                raise ValueError(f"Unsupported add noise mode: {config.add_noise_mode}")
    

        # Get model directory name from enum value
        model_dir_name = get_model_directory_name(config.model)


        unpretranslated_dataset_path = f"tool/dataset/BFCL_v4_multiple{language_tag}{translate_level_tag}{noise_tag}.json"
        ground_truth_path = f"tool/dataset/possible_answer/BFCL_v4_multiple.json"
        # file names to write to or read from if applicable

        pre_translate_output_combined_tags = language_tag + translate_level_tag + pre_translate_tag + noise_tag
        inference_raw_output_combined_tags = language_tag + translate_level_tag + pre_translate_tag + noise_tag + prompt_translate_tag
        post_translate_output_combined_tags = language_tag + translate_level_tag + pre_translate_tag + noise_tag + prompt_translate_tag + post_translate_tag
        
        # assign pre_translate_input_path
        # assign pre_translate_output_path
        if pre_translate_tag == "_pretrans":
            pre_translate_input_path = unpretranslated_dataset_path
            pre_translate_output_path = f"tool/result/pre_translate/{model_dir_name}/{pre_translate_output_combined_tags}.json"
        else:
            assert pre_translate_tag == "_nopretrans"
        # assign inference_raw_input_path
        if pre_translate_tag == "_pretrans":            
            inference_raw_input_path = f"tool/result/pre_translate/{model_dir_name}/{pre_translate_output_combined_tags}.json"
        elif pre_translate_tag == "_nopretrans":            
            inference_raw_input_path = unpretranslated_dataset_path
        else:
            raise ValueError(f"Unsupported pre_translate_tag: {pre_translate_tag}")
        
        
        # assign inference_raw_output_path
        inference_raw_output_path = f"tool/result/inference_raw/{model_dir_name}/{inference_raw_output_combined_tags}.json"
        # assign inference_json_input_path
        inference_json_input_path = inference_raw_output_path
        # assign inference_json_output_path
        inference_json_output_path = f"tool/result/inference_json/{model_dir_name}/{inference_raw_output_combined_tags}.json"
        # assign post_translate_input_path
        post_translate_input_path = inference_json_output_path
        # assign post_translate_output_path
        if post_translate_tag == "_posttrans":
            post_translate_output_path = f"tool/result/post_translate/{model_dir_name}/{post_translate_output_combined_tags}.json"
        else:
            assert post_translate_tag == "_noposttrans"
        # assign evaluation_input_path
        if post_translate_tag == "_posttrans":
            evaluation_input_path = f"tool/result/post_translate/{model_dir_name}/{post_translate_output_combined_tags}.json"
        else:
            evaluation_input_path = post_translate_input_path
        # assign evaluation_output_path
        evaluation_output_path = f"tool/result/evaluation/{model_dir_name}/{post_translate_output_combined_tags}.json"
        # assign score_input_path
        score_input_path = evaluation_output_path
        # assign score_output_path
        score_output_path = f"tool/result/score/{model_dir_name}/{post_translate_output_combined_tags}.json"
        # assign categorize_input_path
        categorize_input_path = score_output_path
        # assign categorize_output_path
        categorize_output_path = f"tool/result/categorize/{model_dir_name}/{post_translate_output_combined_tags}.json"
        # assign categorize_score_input_path
        categorize_score_input_path = categorize_output_path
        # assign categorize_score_output_path
        categorize_score_output_path = f"tool/result/categorize_score/{model_dir_name}/{post_translate_output_combined_tags}.json"

        test_cases = load_json_lines(unpretranslated_dataset_path)
        ground_truths = load_json_lines(ground_truth_path)

        # ═══════════════════════════════════════════════════════════════════════
        # PASS 1: Translated Questions (Pre-Translation)
        # ═══════════════════════════════════════════════════════════════════════
        # Translates questions from the source language to English before inference.
        # This pass runs when FULLY_TRANSLATED_PRE_TRANSLATE option is enabled.
        # Output: tool/result/pre_translate/{model}/{language}.json
        # ═══════════════════════════════════════════════════════════════════════
        if pre_translate_tag == "_nopretrans":
            # Skip translation - pass through original test cases
            print(f"Skipping question translation (pre-translate not enabled)")
        else:
            assert pre_translate_tag == "_pretrans"
            try:
                pre_translate_results, existing_pre_translate_ids = load_json_lines_with_id(pre_translate_output_path)
                existing_pre_translate_ids = {entry["id"] for entry in pre_translate_results}
            except FileNotFoundError:
                print(f"File {pre_translate_output_path} not found. It will be created.")
                pre_translate_results = []
                existing_pre_translate_ids = set()

            # Filter cases that haven't been translated yet
            cases_to_translate = [case for case in test_cases if case['id'] not in existing_pre_translate_ids]

            if len(cases_to_translate) == 0:
                print(f"All test cases have already been translated. Skipping translation.")
            else:
                print(f"Translating {len(cases_to_translate)} questions to English...")

                # Get backend and interface for translation
                translation_backend = get_or_create_backend(
                    model=config.model,
                    num_gpus=args.num_gpus,
                    max_model_len=2000,
                    instance_name="experiment"  # Use experiment instance for pre-translation
                )
                translation_interface = get_or_create_model_interface(config.model)

                async def translate_questions_async():
                    """Translate questions asynchronously."""
                    async def translate_single_question(case):
                        """Translate a single question and return the modified case."""
                        question = case["question"][0][0]['content']

                        # Use the dedicated translation method
                        translated_question = await translation_interface.translate_tool_question_async(
                            backend=translation_backend,
                            question=question
                        )

                        # Create modified case with translated question
                        modified_case = case.copy()
                        modified_case["question"][0][0]['content'] = translated_question

                        return modified_case

                    # Create all translation tasks
                    tasks = [translate_single_question(case) for case in cases_to_translate]

                    # Process results as they complete
                    completed_count = 0
                    for coro in asyncio.as_completed(tasks):
                        modified_case = await coro
                        completed_count += 1

                        print(f"[{completed_count}/{len(cases_to_translate)}] Translated question for case {modified_case['id']}")

                        pre_translate_results.append(modified_case)

                        # Write to file immediately
                        write_json_lines_to_file(pre_translate_output_path, pre_translate_results)

                # Run the async translation
                await translate_questions_async()

                print(f"All {len(cases_to_translate)} questions translated.")

                # Final sort and write
                if len(pre_translate_results) > 0:
                    append_and_rewrite_json_lines(pre_translate_output_path, pre_translate_results)

        # ═══════════════════════════════════════════════════════════════════════
        # PASS 2: Inference Raw
        # ═══════════════════════════════════════════════════════════════════════
        # Generates raw model outputs for each test case using function calling.
        # Input: test_cases (from pre_translate if pre-translate enabled, else dataset)
        # Output: tool/result/inference_raw/{model}/{filename}.json
        # ═══════════════════════════════════════════════════════════════════════
        try:
            inference_json_inputs, existing_inference_ids = load_json_lines_with_id(inference_raw_output_path)
            # Filter out entries with error results
            inference_json_inputs = [
                entry for entry in inference_json_inputs
                if not (isinstance(entry.get("result"), str) and "Error: An error occurred" in entry.get("result", ""))
            ]
            existing_inference_ids = {entry["id"] for entry in inference_json_inputs}
        except FileNotFoundError:
            print(f"File {inference_raw_output_path} not found. It will be created.")
            inference_json_inputs = []
            existing_inference_ids = set()

        printed_warning = False

        # load the input dataset
        preprocessed_test_cases = load_json_lines(inference_raw_input_path)
        # Filter cases that haven't been processed yet
        cases_to_process = [case for case in preprocessed_test_cases if case['id'] not in existing_inference_ids]
        if not printed_warning and len(cases_to_process) < len(preprocessed_test_cases):
            print(f"Warning: some test cases already exist in inference result file. Skipping {len(preprocessed_test_cases) - len(cases_to_process)} cases.")
            printed_warning = True

        # Skip model loading if no cases to process
        if len(cases_to_process) == 0:
            print(f"All test cases for {config.model.value} have already been processed. Skipping model loading and inference.")
        else:
                print("Entering inference phase...")
    
                # Model interface can be created outside async context
                model_interface = get_or_create_model_interface(config.model)
    
                # Process requests asynchronously
                print(f"\nSubmitting {len(cases_to_process)} requests concurrently...")
                if prompt_translate_tag == "_prompt":
                    prompt_translate = True
                else:
                    assert prompt_translate_tag == "_noprompt"
                    prompt_translate = False
    
                async def process_batch_async():
                    """Process batch requests asynchronously."""
                    # Create backend inside async context to ensure it's tied to the current event loop
                    backend = get_or_create_backend(
                        model=config.model,
                        num_gpus=args.num_gpus,
                        max_model_len=2000,
                        instance_name="experiment"  # Use experiment instance for inference
                    )
    
                    async def process_single_case(case):
                        """Process a single case and return the result with case info."""
                        functions = case['function']
                        user_question = case["question"][0][0]['content']
                        result = await model_interface.generate_tool_call_async(
                            backend=backend,
                            raw_functions=functions,
                            user_query=user_question,
                            name_mapper=get_global_name_mapper(),
                            prompt_passing_in_english=prompt_translate
                        )
                        return case, result
    
                    # Create all tasks
                    tasks = [process_single_case(case) for case in cases_to_process]
    
                    # Process results as they complete
                    completed_count = 0
                    for coro in asyncio.as_completed(tasks):
                        case, result = await coro
                        completed_count += 1
    
                        print(f"[{completed_count}/{len(cases_to_process)}] Case {case['id']}: {case['question'][0][0]['content'][:60]}...")
    
                        result_to_write = {
                            "id": case["id"],
                            "result": result
                        }
                        inference_json_inputs.append(result_to_write)
    
                        # Write to file immediately (unsorted)
                        write_json_lines_to_file(inference_raw_output_path, inference_json_inputs)
    
                # Run the async batch processing (now using await since we're in async context)
                await process_batch_async()
    
                print(f"All {len(cases_to_process)} requests completed.")
    
                # Final sort and write
                if len(inference_json_inputs) > 0:
                    append_and_rewrite_json_lines(inference_raw_output_path, inference_json_inputs)
    
        # Populate global name mapper if this model requires name sanitization
        # This is done once per model, independent of whether we have a model_interface
        # This is much cheaper than loading the model just to build name mappings
        if requires_name_sanitization(config.model):
            # Collect all unique functions from test_cases
            all_functions = []
            seen_functions = set()
            for case in preprocessed_test_cases:
                for func in case['function']:
                    func_name = func.get('name')
                    if func_name and func_name not in seen_functions:
                        all_functions.append(func)
                        seen_functions.add(func_name)

            # Populate the global name mapper (model-agnostic, no model loading needed)
            name_mapper = get_global_name_mapper()
            name_mapper.populate_from_functions(all_functions)
        else:
            name_mapper = None

        # Ensure model_interface is created before inference_json or other passes
        model_interface = get_or_create_model_interface(config.model)

        # ═══════════════════════════════════════════════════════════════════════
        # PASS 3: Inference JSON
        # ═══════════════════════════════════════════════════════════════════════
        # Parses and postprocesses raw model outputs into structured JSON format.
        # Converts model-specific output format to standardized function call format.
        # Output: tool/result/inference_json/{model}/{filename}.json
        # ═══════════════════════════════════════════════════════════════════════
        # reload inference raw results
        try:
            inference_json_inputs = load_json_lines(inference_json_input_path)
        except FileNotFoundError:
            print(f"Error: File {inference_json_input_path} not found.")
            exit(1)

        inference_json_results = []

        for inference_raw in inference_json_inputs:
            id = inference_raw['id']
            # convert raw result to json format
            # decoded_output = raw_to_json(config.model, id, inference_raw['result'])
            # Pass name_mapper to parse_output for models that need name sanitization
            result = model_interface.postprocess_tool_calls(inference_raw['result'], name_mapper=name_mapper)

            # Check if result is an error tuple or success list
            if isinstance(result, tuple):
                # Error case: (PostprocessError, metadata_dict)
                error, metadata = result
                inference_json_entry = {
                    "id": id,
                    "valid": False,
                    "error": error.value,
                    "error_meta": metadata
                }
            else:
                # Success case: list of function calls
                assert isinstance(result, list)
                inference_json_entry = {
                    "id": id,
                    "valid": True,
                    "result": result
                }
            inference_json_results.append(inference_json_entry)

            # Write batch results to file
            write_json_lines_to_file(inference_json_output_path, inference_json_results)

        # Final sort and write
        if len(inference_json_results) > 0:
            append_and_rewrite_json_lines(inference_json_output_path, inference_json_results)
        # ═══════════════════════════════════════════════════════════════════════
        # PASS 4: Translated Answers (Post-Translation)
        # ═══════════════════════════════════════════════════════════════════════
        # Translates function call parameter values from source language to English.
        # This pass runs when FULLY_TRANSLATED_POST_TRANSLATE option is enabled.
        # Input: tool/result/inference_json/{model}/{filename}.json
        # Output: tool/result/translated_answers/{model}/{language}.json
        # ═══════════════════════════════════════════════════════════════════════
        if post_translate_tag == "_noposttrans":
            # Skip translation - pass through original inference json results
            print(f"Skipping answer translation (post-translate not enabled)")
        else:
            assert post_translate_tag == "_posttrans"
            # Load inference json results
            try:
                inference_json_results = load_json_lines(post_translate_input_path)
            except FileNotFoundError:
                print(f"Error: File {post_translate_input_path} not found.")
                exit(1)

            try:
                translated_answers_results, existing_translated_answers_ids = load_json_lines_with_id(post_translate_output_path)
                existing_translated_answers_ids = {entry["id"] for entry in translated_answers_results}
            except FileNotFoundError:
                print(f"File {post_translate_output_path} not found. It will be created.")
                translated_answers_results = []
                existing_translated_answers_ids = set()

            # Filter samples that haven't been translated yet
            samples_to_translate = [sample for sample in inference_json_results if sample['id'] not in existing_translated_answers_ids]

            if len(samples_to_translate) == 0:
                print(f"All answers have already been translated. Skipping translation.")
            else:
                print(f"Translating {len(samples_to_translate)} answers to English...")

                # Get backend and interface for translation
                translation_backend = get_or_create_backend(
                    model=config.model,
                    num_gpus=args.num_gpus,
                    max_model_len=2000,
                    instance_name="experiment"  # Use experiment instance for post-translation
                )
                translation_interface = get_or_create_model_interface(config.model)

                async def translate_answers_async():
                    """Translate function call parameters asynchronously."""
                    async def translate_list_values(items: list) -> list:
                        """
                        Recursively translate string values within a list.

                        For example:
                        ["鸡肉", "蘑菇"] -> ["chicken", "mushroom"]
                        """
                        # Collect all items that need translation
                        translation_tasks = []
                        indices_for_strings = []

                        for i, item in enumerate(items):
                            if isinstance(item, str) and item.strip():
                                # Translate string items
                                translation_tasks.append(
                                    translation_interface.translate_tool_answer_async(
                                        backend=translation_backend,
                                        parameter_value=item
                                    )
                                )
                                indices_for_strings.append(i)

                        # Create result list with original items
                        translated_list = list(items)

                        # Wait for all string translations to complete
                        if translation_tasks:
                            translated_values = await asyncio.gather(*translation_tasks)
                            # Replace translated strings at their original indices
                            for idx, translated_value in zip(indices_for_strings, translated_values):
                                translated_list[idx] = translated_value

                        # Second pass: recursively translate nested dicts and lists
                        for i, item in enumerate(translated_list):
                            if isinstance(item, dict):
                                translated_list[i] = await translate_dict_values(item)
                            elif isinstance(item, list):
                                translated_list[i] = await translate_list_values(item)

                        return translated_list

                    async def translate_dict_values(arguments: dict) -> dict:
                        """
                        Recursively translate only the VALUES in a dictionary, preserving all KEYS unchanged.

                        For example:
                        {"location": "北京"} -> {"location": "Beijing"}  # Key preserved, value translated
                        """
                        translated = {}

                        # First pass: collect all string values that need translation
                        # IMPORTANT: We only translate VALUES, never KEYS (parameter names)
                        translation_tasks = []
                        keys_for_string_values = []  # Store parameter names (not translated)

                        for param_name, param_value in arguments.items():
                            if isinstance(param_value, str) and param_value.strip():
                                # Translate this string VALUE
                                # The parameter NAME (param_name) is preserved as-is
                                translation_tasks.append(
                                    translation_interface.translate_tool_answer_async(
                                        backend=translation_backend,
                                        parameter_value=param_value
                                    )
                                )
                                keys_for_string_values.append(param_name)
                            elif isinstance(param_value, (dict, list)):
                                # Skip for now - will handle in second pass
                                pass
                            else:
                                # Keep non-string values as-is (numbers, booleans, etc.)
                                translated[param_name] = param_value

                        # Wait for all string value translations to complete
                        if translation_tasks:
                            translated_values = await asyncio.gather(*translation_tasks)
                            # Map translated values back to their ORIGINAL parameter names
                            for param_name, translated_value in zip(keys_for_string_values, translated_values):
                                translated[param_name] = translated_value

                        # Second pass: recursively translate nested dictionaries and lists
                        # Again, only the values in nested dicts/lists are translated, not the keys
                        for param_name, param_value in arguments.items():
                            if isinstance(param_value, dict):
                                translated[param_name] = await translate_dict_values(param_value)
                            elif isinstance(param_value, list):
                                translated[param_name] = await translate_list_values(param_value)

                        return translated

                    async def translate_single_answer(sample):
                        """Translate parameters in a single sample and return the modified sample."""
                        # If sample is invalid (postprocess error), return as-is
                        if not sample.get("valid", True):  # Default to True for backward compatibility
                            return sample

                        result = sample.get("result", [])

                        # If result is not a list or is empty, return as is
                        if not isinstance(result, list) or len(result) == 0:
                            return sample

                        modified_result = []
                        for func_call in result:
                            if not isinstance(func_call, dict):
                                modified_result.append(func_call)
                                continue

                            # Get the function name and arguments
                            func_name = list(func_call.keys())[0] if func_call else None
                            if not func_name:
                                modified_result.append(func_call)
                                continue

                            arguments = func_call.get(func_name, {})

                            # If no arguments, skip translation
                            if not arguments or not isinstance(arguments, dict):
                                modified_result.append(func_call)
                                continue

                            # Translate all string values in arguments
                            try:
                                translated_arguments = await translate_dict_values(arguments)

                                # Create modified function call with translated arguments
                                modified_result.append({func_name: translated_arguments})
                            except Exception as e:
                                print(f"Error: Failed to translate parameters for sample {sample['id']}: {e}")
                                exit(1)
                                # # Keep original if translation fails
                                # modified_result.append(func_call)

                        # Create modified sample with translated parameters
                        modified_sample = sample.copy()
                        modified_sample["result"] = modified_result

                        return modified_sample

                    # Create all translation tasks
                    tasks = [translate_single_answer(sample) for sample in samples_to_translate]

                    # Process results as they complete
                    completed_count = 0
                    for coro in asyncio.as_completed(tasks):
                        modified_sample = await coro
                        completed_count += 1

                        print(f"[{completed_count}/{len(samples_to_translate)}] Translated answer parameters for sample {modified_sample['id']}")

                        translated_answers_results.append(modified_sample)

                        # Write to file immediately
                        write_json_lines_to_file(post_translate_output_path, translated_answers_results)

                # Run the async translation
                await translate_answers_async()

                print(f"All {len(samples_to_translate)} answers translated.")

                # Final sort and write
                if len(translated_answers_results) > 0:
                    append_and_rewrite_json_lines(post_translate_output_path, translated_answers_results)

        # ═══════════════════════════════════════════════════════════════════════
        # PASS 5: Evaluation
        # ═══════════════════════════════════════════════════════════════════════
        # Evaluates model outputs against ground truth to determine correctness.
        # Checks function names, parameter names, and parameter values.
        # Input: tool/result/post_translate if post-translate enabled, else inference_json
        # Output: tool/result/evaluation/{model}/{filename}.json
        # ═══════════════════════════════════════════════════════════════════════
        # reload inference results for evaluation
        try:
            inference_results = load_json_lines(evaluation_input_path)
        except FileNotFoundError:
            print(f"File {evaluation_input_path} not found. Skipping evaluation.")
            exit(1)
        evaluation_results = []

        for (inference_line, ground_truth_line, test_case) in zip(inference_results, ground_truths, test_cases):
            id = inference_line["id"]
            assert id == ground_truth_line["id"], f"Mismatch in IDs: {id} vs {ground_truth_line['id']}"
            assert id == test_case["id"], f"Mismatch in IDs: {id} vs {test_case['id']}"

            # Check if postprocess result was valid
            if inference_line.get("valid", True):  # Default to True for backward compatibility
                # Valid result: evaluate normally
                inference_result = inference_line["result"]
                ground_truth = ground_truth_line["ground_truth"]
                func_description = test_case['function']

                eval_result = evaluate_json(id, inference_result, ground_truth, func_description)

                # Check if evaluation returned error or success
                if isinstance(eval_result, tuple):
                    # Evaluation error
                    error, metadata = eval_result
                    evaluation_entry = {
                        "id": id,
                        "valid": False,
                        "error": error.value,
                        "error_meta": metadata
                    }
                else:
                    # Evaluation success
                    assert isinstance(eval_result, dict)
                    evaluation_entry = eval_result
            else:
                # Invalid result from postprocess: pass through the error
                evaluation_entry = {
                    "id": id,
                    "valid": False,
                    "error": inference_line.get('error', 'unknown'),
                    "error_meta": inference_line.get("error_meta", {})
                }

            evaluation_results.append(evaluation_entry)

            # Write batch results to file
            write_json_lines_to_file(evaluation_output_path, evaluation_results)

        # Final sort and write
        if len(evaluation_results) > 0:
            append_and_rewrite_json_lines(evaluation_output_path, evaluation_results)
        # ═══════════════════════════════════════════════════════════════════════
        # PASS 6: Score
        # ═══════════════════════════════════════════════════════════════════════
        # Calculates accuracy and aggregates wrong cases for analysis.
        # Input: tool/result/evaluation/{model}/{filename}.json
        # Output: tool/result/score/{model}/{filename}.json
        # ═══════════════════════════════════════════════════════════════════════
        # reload evaluation results
        try:
            evaluation_entries = load_json_lines(score_input_path)
        except FileNotFoundError:
            print(f"File {score_input_path} not found. Skipping scoring.")
            exit(1)
        # Calculate and write score results
        total_cases = 0
        correct_cases = 0
        wrong_cases = []
        score_results = []

        for evaluation_entry in evaluation_entries:
            total_cases += 1
            if evaluation_entry['valid']:
                correct_cases += 1
            else:
                wrong_cases.append(evaluation_entry)

        accuracy = correct_cases / total_cases if total_cases > 0 else 0.0
        # Add summary score
        score_result = {
            "accuracy": accuracy,
            "total_cases": total_cases,
            "correct_cases": correct_cases,
        }
        score_results.append(score_result)

        # Add wrong cases
        score_results.extend(wrong_cases)

        # Write all results to file
        write_json_lines_to_file(score_output_path, score_results)
        print(f"Score result written to {score_output_path}: {score_result}")

        # ═══════════════════════════════════════════════════════════════════════
        # PASS 7: Categorize
        # ═══════════════════════════════════════════════════════════════════════
        # Categorizes each error sample into different error types.
        # Input: tool/result/score/{model}/{filename}.json
        # Output: tool/result/categorize/{model}/{filename}.json
        # ═══════════════════════════════════════════════════════════════════════
        # Load error samples from score file
        try:
            score_entries = load_json_lines(categorize_input_path)
        except FileNotFoundError:
            print(f"File {categorize_input_path} not found. Skipping categorization.")
            score_entries = []

        # Filter out the summary entry (first line in score file)
        samples_to_categorize = [entry for entry in score_entries if 'accuracy' not in entry]

        if len(samples_to_categorize) == 0:
            print(f"No error samples found. Skipping categorization.")
        else:
            print(f"Categorizing {len(samples_to_categorize)} error samples...")

            # Acquire cache lock for entire categorize pass
            print("Acquiring cache lock...")
            cache_lock.acquire()
            print("Acquired cache lock")

            # Load category cache
            category_cache = load_category_cache(category_cache_path)

            categorize_results = []
            cache_hits = 0
            cache_misses = 0

            async def categorize_samples_async():
                """Categorize error samples asynchronously."""
                nonlocal cache_hits, cache_misses

                async def categorize_with_sample(sample):
                    """Wrapper to return sample, category, and cache hit status."""
                    category_enum, cache_hit = await categorize_single_sample_async(sample, category_cache)
                    return sample, category_enum, cache_hit

                # Create all categorization tasks
                tasks = [categorize_with_sample(sample) for sample in samples_to_categorize]

                # Process results as they complete
                completed_count = 0
                for coro in asyncio.as_completed(tasks):
                    sample, category_enum, cache_hit = await coro
                    completed_count += 1

                    # Track cache statistics
                    if cache_hit:
                        cache_hits += 1
                    else:
                        cache_misses += 1

                    # Assemble the result dict (assembly logic in tool_main.py)
                    categorized_sample = {
                        "id": sample["id"],
                        "category": category_enum.value,  # Store enum value as string
                        "evaluation_entry": sample
                    }

                    # Only print on cache miss, and only show parameter comparison details
                    if not cache_hit:
                        error_meta = sample.get("error_meta", {})
                        actual_value = error_meta.get("actual_value")
                        expected_values = error_meta.get("expected_values", [])
                        if actual_value is not None and expected_values:
                            print(f"[{completed_count}/{len(samples_to_categorize)}] Comparing actual: {json.dumps(actual_value, ensure_ascii=False)} with expected: {json.dumps(expected_values, ensure_ascii=False)} -> {category_enum.value}")

                    categorize_results.append(categorized_sample)

                    # Write to file immediately
                    write_json_lines_to_file(categorize_output_path, categorize_results)

            try:
                # Run the async categorization
                await categorize_samples_async()

                print(f"All {len(samples_to_categorize)} samples categorized.")
                print(f"Cache statistics - Hits: {cache_hits}, Misses: {cache_misses}, Hit rate: {cache_hits / (cache_hits + cache_misses) * 100:.2f}%" if (cache_hits + cache_misses) > 0 else "Cache statistics - No cache lookups performed")

                # Final sort and write
                if len(categorize_results) > 0:
                    append_and_rewrite_json_lines(categorize_output_path, categorize_results)

                # Write cache back to file once at the end
                save_category_cache(category_cache_path, category_cache)
            finally:
                # Release lock at the end of categorize pass
                cache_lock.release()
                print("Released cache lock")

            # Destroy local cache container (Python will handle this automatically when it goes out of scope)
            del category_cache

        # ═══════════════════════════════════════════════════════════════════════
        # PASS 8: Categorize Score
        # ═══════════════════════════════════════════════════════════════════════
        # Aggregates categorization results and counts errors for each category.
        # Lightweight pass that always overwrites existing file.
        # Input: tool/result/categorize/{model}/{filename}.json
        # Output: tool/result/categorize_score/{model}/{filename}.json
        # ═══════════════════════════════════════════════════════════════════════
        # Load categorized results
        try:
            categorized_samples = load_json_lines(categorize_score_input_path)
        except FileNotFoundError:
            print(f"File {categorize_score_input_path} not found. Skipping categorize scoring.")
            categorized_samples = []

        if len(categorized_samples) == 0:
            print(f"No categorized samples found. Skipping categorize scoring.")
        else:
            # Aggregate statistics by category
            category_counts = {}
            category_samples = {}
            for result in categorized_samples:
                category = result['category']
                if category not in category_counts:
                    category_counts[category] = 0
                    category_samples[category] = []
                category_counts[category] += 1
                category_samples[category].append(result['id'])

            # Prepare final output with summary and samples
            final_output = {
                "summary": category_counts,
                "samples": category_samples
            }

            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(categorize_score_output_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # Write final aggregated results (single write, overwrites existing)
            with open(categorize_score_output_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)

            print(f"Categorize score results written to {categorize_score_output_path}")
            print("\nError Category Summary:")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {category}: {count}")

        print(f"Completed processing for config: {config}")
            # Run all configs in a single event loop
asyncio.run(process_all_configs())
