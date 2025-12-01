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
from tool.post_processing import (
    load_or_create_cache,
    save_cache,
    process_post_processing_sample
)
from models.name_mapping import get_global_name_mapper

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

def load_json_lines_from_file(file_path: str) -> tuple[list, set]:
    """
    Load JSON lines from a file and extract existing IDs.

    Args:
        file_path: Path to the JSON lines file

    Returns:
        Tuple of (results_list, existing_ids_set)
    """
    results = []
    existing_ids = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if f.readable():
                for line in f:
                    line_json = json.loads(line)
                    id = line_json["id"]
                    results.append(line_json)
                    existing_ids.add(id)
    except FileNotFoundError:
        print(f"File {file_path} not found. It will be created.")
    return results, existing_ids


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


def get_result_filename(language_postfix: str, mode_postfix: str, noise_postfix: str) -> str:
    """
    Construct result filename from postfix components.

    Args:
        language_postfix: Language postfix (e.g., "_zh", "_hi", or "")
        mode_postfix: Translate/processing mode postfix (e.g., "_f", "_pt", "_par", or "")
        noise_postfix: Noise mode postfix (e.g., "_syno", "_para", or "")

    Returns:
        Filename for the result file
    """
    # Combine all postfixes
    combined = language_postfix + mode_postfix + noise_postfix

    # If no postfixes, this is the vanilla/baseline case
    if not combined:
        return "vanilla.json"

    # Remove leading underscore and add .json extension
    filename = combined.lstrip("_") + ".json"
    return filename


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

# Run inference
# Global variables to track if generator (pipeline/client) and model_interface are initialized (reuse across configs)
_global_generator = None  # For local models: pipeline; for API models: client
_global_model_interface = None
_global_model = None
_global_backend = None  # Only relevant for local models (vLLM vs HuggingFace)



def get_or_create_backend(model: Model, num_gpus: int = 1, max_model_len: int = 2000):
    """
    Get or create a backend for the given model.

    Args:
        model: ApiModel or LocalModel enum value
        num_gpus: Number of GPUs to use
        max_model_len: Maximum model length

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
        max_model_len=max_model_len
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


# Global caches for post-processing parameter matching (shared across all configs)
# Separate caches for different language handling options
post_processing_cache_different_path = "tool/post_processing_match_cache_different.json"
post_processing_cache_same_path = "tool/post_processing_match_cache_same.json"
post_processing_cache_different = load_or_create_cache(post_processing_cache_different_path)
post_processing_cache_same = load_or_create_cache(post_processing_cache_same_path)
post_processing_cache_stats_different = {'hits': 0, 'misses': 0}
post_processing_cache_stats_same = {'hits': 0, 'misses': 0}


async def process_all_configs():
    """Process all configs within a single event loop to allow backend reuse."""
    for config in configs:
        print(f"Processing config: {config}", flush=True)

        post_process_option = PostProcessOption.DONT_POST_PROCESS
        prompt_translate = False
        # map translate_info to language_postfix, translate_dataset_prefix, translate_mode_prefix
        match config.translate_mode:
            case Translated(language, option):
                match language:
                    case Language.CHINESE:
                        language_postfix = "_zh"
                    case Language.HINDI:
                        language_postfix = "_hi"
                match option:
                    case TranslateOption.FULLY_TRANSLATED:
                        translate_dataset_postfix = "_full"
                        translate_mode_postfix = "_f" # fully translated, default
                        translate_postfix = "_f" # fully translated, do not prompt translate
                    case TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE:
                        translate_dataset_postfix = "_full"
                        translate_mode_postfix = "_pt"  # prompt translate
                        translate_postfix = "_fp" # fully translated, prompt translate
                        prompt_translate = True
                    case TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT:
                        translate_dataset_postfix = "_full"
                        translate_mode_postfix = "_ppd"  # post-process different
                        translate_postfix = "_f" # fully translated, do not prompt translate
                        post_process_option = PostProcessOption.POST_PROCESS_DIFFERENT
                    case TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME:
                        translate_dataset_postfix = "_full"
                        translate_mode_postfix = "_pps"  # post-process same
                        translate_postfix = "_f" # fully translated, do not prompt translate
                        post_process_option = PostProcessOption.POST_PROCESS_SAME
                    case TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE_POST_PROCESS_SAME:
                        translate_dataset_postfix = "_full"
                        translate_mode_postfix = "_ptps"  # prompt translate + post-process same
                        translate_postfix = "_fp" # fully translated, prompt translate
                        post_process_option = PostProcessOption.POST_PROCESS_SAME
                    case TranslateOption.PARTIALLY_TRANSLATED:
                        translate_dataset_postfix = "_partial"
                        translate_mode_postfix = "_par" # partial
                        translate_postfix = "_par" # partially translated, do not prompt translate
                    case _:
                        raise ValueError(f"Unsupported translate option: {option}")
            case NotTranslated():
                language_postfix = ""
                translate_dataset_postfix = ""
                translate_mode_postfix = ""
                translate_postfix = ""
        match config.add_noise_mode:
            case AddNoiseMode.NO_NOISE:
                noise_postfix = ""
            case AddNoiseMode.SYNONYM:
                noise_postfix = "_syno"
            case AddNoiseMode.PARAPHRASE:
                noise_postfix = "_para"
            case _:
                raise ValueError(f"Unsupported add noise mode: {config.add_noise_mode}")
    

        # Get model directory name from enum value
        model_dir_name = get_model_directory_name(config.model)

        # Construct filenames based on configuration
        inference_filename = get_result_filename(language_postfix, translate_postfix, noise_postfix)
        processing_filename = get_result_filename(language_postfix, translate_mode_postfix, noise_postfix)

        dataset_path = f"tool/dataset/BFCL_v4_multiple{language_postfix}{translate_dataset_postfix}{noise_postfix}.json"
        ground_truth_path = f"tool/dataset/possible_answer/BFCL_v4_multiple.json"
        inference_raw_result_path = f"tool/result/inference_raw/{model_dir_name}/{inference_filename}"
        inference_json_result_path = f"tool/result/inference_json/{model_dir_name}/{inference_filename}"
        post_processing_result_path = f"tool/result/post_processing/{model_dir_name}/{processing_filename}"
        evaluation_result_path = f"tool/result/evaluation/{model_dir_name}/{processing_filename}"
        score_path = f"tool/result/score/{model_dir_name}/{processing_filename}"

        test_cases, _ = load_json_lines_from_file(dataset_path)
        ground_truths, _ = load_json_lines_from_file(ground_truth_path)

        if requires_inference_raw:
            try:
                inference_raw_results, existing_inference_ids = load_json_lines_from_file(inference_raw_result_path)
                # Filter out entries with error results
                inference_raw_results = [
                    entry for entry in inference_raw_results
                    if not (isinstance(entry.get("result"), str) and "Error: An error occurred" in entry.get("result", ""))
                ]
                existing_inference_ids = {entry["id"] for entry in inference_raw_results}
            except FileNotFoundError:
                print(f"File {inference_raw_result_path} not found. It will be created.")
                inference_raw_results = []
                existing_inference_ids = set()
        
            printed_warning = False
            # Filter cases that haven't been processed yet
            cases_to_process = [case for case in test_cases if case['id'] not in existing_inference_ids]
            if not printed_warning and len(cases_to_process) < len(test_cases):
                print(f"Warning: some test cases already exist in inference result file. Skipping {len(test_cases) - len(cases_to_process)} cases.")
                printed_warning = True
    
            # Skip model loading if no cases to process
            if len(cases_to_process) == 0:
                print(f"All test cases for {config.model.value} have already been processed. Skipping model loading and inference.")
            else:
                print("Entering inference phase...")
                # Determine model type and create interface once
                is_api_model = isinstance(config.model, ApiModel)
                is_local_model = isinstance(config.model, LocalModel)
    
                # Configure concurrent request settings
                if is_api_model:
                    max_concurrent = 8  # API models: up to 8 concurrent requests
                    print(f"API model inference configuration:")
                    print(f"  Model: {config.model.value}")
                    print(f"  Max concurrent requests: {max_concurrent}")
                else:
                    # For local models with vLLM, we can submit all requests concurrently
                    # vLLM's engine will handle internal batching automatically
                    max_concurrent = len(cases_to_process)  # Submit all at once
                    model_size_b = extract_model_size_in_billions(config.model)
                    print(f"Local model inference configuration:")
                    print(f"  Model: {config.model.value}")
                    print(f"  Model size: {model_size_b}B")
                    print(f"  Number of GPUs: {args.num_gpus}")
                    print(f"  Backend: {'vLLM' if USE_VLLM_BACKEND else 'HuggingFace'}")
                    print(f"  Concurrent requests: {max_concurrent} (vLLM handles internal batching)")
    
                # Model interface can be created outside async context
                model_interface = get_or_create_model_interface(config.model)
    
                # Process requests asynchronously
                print(f"\nSubmitting {len(cases_to_process)} requests concurrently...")
    
                async def process_batch_async():
                    """Process batch requests asynchronously."""
                    # Create backend inside async context to ensure it's tied to the current event loop
                    backend = get_or_create_backend(
                        model=config.model,
                        num_gpus=args.num_gpus,
                        max_model_len=2000
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
                        inference_raw_results.append(result_to_write)
    
                        # Write to file immediately (unsorted)
                        write_json_lines_to_file(inference_raw_result_path, inference_raw_results)
    
                # Run the async batch processing (now using await since we're in async context)
                await process_batch_async()
    
                print(f"All {len(cases_to_process)} requests completed.")
    
                # Final sort and write
                if len(inference_raw_results) > 0:
                    append_and_rewrite_json_lines(inference_raw_result_path, inference_raw_results)
    
        # Populate global name mapper if this model requires name sanitization
        # This is done once per model, independent of whether we have a model_interface
        # This is much cheaper than loading the model just to build name mappings
        if requires_name_sanitization(config.model):
            # Collect all unique functions from test_cases
            all_functions = []
            seen_functions = set()
            for case in test_cases:
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

        if requires_inference_json:
        # reload inference raw results
            try:
                inference_raw_results, _ = load_json_lines_from_file(inference_raw_result_path)
            except FileNotFoundError:
                print(f"File {inference_raw_result_path} not found. Skipping inference json generation.")
                continue
            if evaluation_caching:
                try:
                    inference_json_results, existing_inference_json_ids = load_json_lines_from_file(inference_json_result_path)
                except FileNotFoundError:
                    print(f"File {inference_json_result_path} not found. Skipping inference json caching.")
                    inference_json_results = []
                    existing_inference_json_ids = set()
            else:
                inference_json_results = []
                existing_inference_json_ids = set()
            printed_warning = False
            # Filter samples that haven't been processed yet
            samples_to_process = [sample for sample in inference_raw_results if sample['id'] not in existing_inference_json_ids]
            if not printed_warning and len(samples_to_process) < len(inference_raw_results):
                print(f"Warning: some test cases already exist in inference json result file. Skipping {len(inference_raw_results) - len(samples_to_process)} cases.")
                printed_warning = True
    
            for inference_raw in samples_to_process:
                id = inference_raw['id']
                # convert raw result to json format
                # decoded_output = raw_to_json(config.model, id, inference_raw['result'])
                # Pass name_mapper to parse_output for models that need name sanitization
                decoded_output = model_interface.postprocess_tool_calls(inference_raw['result'], name_mapper=name_mapper)
                inference_json_entry = {
                    "id": id,
                    "result": decoded_output
                }
                inference_json_results.append(inference_json_entry)
    
                # Write batch results to file
                write_json_lines_to_file(inference_json_result_path, inference_json_results)
    
            # Final sort and write
            if len(inference_json_results) > 0:
                append_and_rewrite_json_lines(inference_json_result_path, inference_json_results)

        if requires_post_processing:
            if post_process_option == PostProcessOption.DONT_POST_PROCESS:
                # Simply copy inference_json results to post_processing results without modification
                try:
                    inference_json_results, _ = load_json_lines_from_file(inference_json_result_path)
                except FileNotFoundError:
                    print(f"File {inference_json_result_path} not found. Skipping post processing.")
                    continue
    
                if evaluation_caching:
                    try:
                        post_processing_results, existing_post_processing_ids = load_json_lines_from_file(post_processing_result_path)
                    except FileNotFoundError:
                        print(f"File {post_processing_result_path} not found. Skipping post processing caching.")
                        post_processing_results = []
                        existing_post_processing_ids = set()
                else:
                    post_processing_results = []
                    existing_post_processing_ids = set()
    
                # Copy unprocessed results
                for inference_json_line in inference_json_results:
                    if inference_json_line['id'] not in existing_post_processing_ids:
                        post_processing_results.append(inference_json_line)
    
                # Final sort and write
                if len(post_processing_results) > 0:
                    append_and_rewrite_json_lines(post_processing_result_path, post_processing_results)
    
                print(f"Post-processing: Copied {len(inference_json_results)} results without modification (DONT_POST_PROCESS)")
    
            else:
                # POST_PROCESS_DIFFERENT or POST_PROCESS_SAME: use LLM-based parameter matching
                # Select appropriate cache based on post_process_option
                if post_process_option == PostProcessOption.POST_PROCESS_SAME:
                    post_processing_cache = post_processing_cache_same
                    post_processing_cache_stats = post_processing_cache_stats_same
                    cache_path = post_processing_cache_same_path
                elif post_process_option == PostProcessOption.POST_PROCESS_DIFFERENT:  # POST_PROCESS_DIFFERENT
                    post_processing_cache = post_processing_cache_different
                    post_processing_cache_stats = post_processing_cache_stats_different
                    cache_path = post_processing_cache_different_path
                else:
                    raise ValueError(f"Unsupported post process option: {post_process_option}")
    
                # reload inference json results
                try:
                    inference_json_results, _ = load_json_lines_from_file(inference_json_result_path)
                except FileNotFoundError:
                    print(f"File {inference_json_result_path} not found. Skipping post processing.")
                    continue
    
                if evaluation_caching:
                    try:
                        post_processing_results, existing_post_processing_ids = load_json_lines_from_file(post_processing_result_path)
                    except FileNotFoundError:
                        print(f"File {post_processing_result_path} not found. Skipping post processing caching.")
                        post_processing_results = []
                        existing_post_processing_ids = set()
                else:
                    post_processing_results = []
                    existing_post_processing_ids = set()
    
                printed_warning = False
                # Filter samples that haven't been processed yet
                samples_to_process = [sample for sample in inference_json_results if sample['id'] not in existing_post_processing_ids]
                if not printed_warning and len(samples_to_process) < len(inference_json_results):
                    print(f"Warning: some test cases already exist in post processing result file. Skipping {len(inference_json_results) - len(samples_to_process)} cases.")
                    printed_warning = True
    
                for inference_json_line in samples_to_process:
                    id = inference_json_line['id']
                    # print(f"Post-processing case id {id}...")
                    # Find matching ground truth
                    ground_truth_line = next((gt for gt in ground_truths if gt['id'] == id), None)
                    if ground_truth_line is None:
                        raise ValueError(f"Ground truth not found for id: {id}")
                    # Process with LLM-based parameter matching
                    post_processing_entry = process_post_processing_sample(
                        inference_json_line,
                        ground_truth_line,
                        ApiModel.GPT_5_MINI,  # Use a powerful model for post-processing
                        post_process_option,
                        post_processing_cache,
                        cache_path,
                        post_processing_cache_stats
                    )
                    post_processing_results.append(post_processing_entry)
    
                    # Write batch results to file
                    write_json_lines_to_file(post_processing_result_path, post_processing_results)
    
                # Final sort and write
                if len(post_processing_results) > 0:
                    append_and_rewrite_json_lines(post_processing_result_path, post_processing_results)
    
                print(f"Post-processing ({post_process_option.name}) completed - Hits: {post_processing_cache_stats['hits']}, Misses: {post_processing_cache_stats['misses']}")
        if requires_evaluation:
            # reload post processing results
            try:
                post_processing_results, _ = load_json_lines_from_file(post_processing_result_path)
            except FileNotFoundError:
                print(f"File {post_processing_result_path} not found. Skipping evaluation.")
                continue
            if evaluation_caching:
                try:
                    evaluation_results, existing_evaluation_ids = load_json_lines_from_file(evaluation_result_path)
                except FileNotFoundError:
                    print(f"File {evaluation_result_path} not found. Skipping evaluation caching.")
                    evaluation_results = []
                    existing_evaluation_ids = set()
            else:
                evaluation_results = []
                existing_evaluation_ids = set()
            printed_warning = False
            # Filter samples that haven't been processed yet
            samples_to_process = [
                (post_processing_line, ground_truth_line, test_case)
                for post_processing_line, ground_truth_line, test_case in zip(post_processing_results, ground_truths, test_cases)
                if post_processing_line["id"] not in existing_evaluation_ids
            ]
            if not printed_warning and len(samples_to_process) < len(post_processing_results):
                print(f"Warning: some test cases already exist in evaluation result file. Skipping {len(post_processing_results) - len(samples_to_process)} cases.")
                printed_warning = True
    
            for (post_processing_line, ground_truth_line, test_case) in samples_to_process:
                id = post_processing_line["id"]
                assert id == ground_truth_line["id"], f"Mismatch in IDs: {id} vs {ground_truth_line['id']}"
                assert id == test_case["id"], f"Mismatch in IDs: {id} vs {test_case['id']}"
                post_processing_result = post_processing_line["result"]
                ground_truth = ground_truth_line["ground_truth"]
                func_description = test_case['function']
    
                evaluation_result = evaluate_json(id, post_processing_result, ground_truth, func_description)
                evaluation_result["id"] = id
                evaluation_results.append(evaluation_result)
    
                # Write batch results to file
                write_json_lines_to_file(evaluation_result_path, evaluation_results)
    
            # Final sort and write
            if len(evaluation_results) > 0:
                append_and_rewrite_json_lines(evaluation_result_path, evaluation_results)
        if requires_score:
            # reload evaluation results
            try:
                evaluation_entries, _ = load_json_lines_from_file(evaluation_result_path)
            except FileNotFoundError:
                print(f"File {evaluation_result_path} not found. Skipping scoring.")
                continue
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
            write_json_lines_to_file(score_path, score_results)
            print(f"Score result written to {score_path}: {score_result}")
            print(f"Completed processing for config: {config}")
            # Run all configs in a single event loop
asyncio.run(process_all_configs())
