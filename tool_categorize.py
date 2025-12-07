"""
Categorization module for analyzing and grouping evaluation errors by type.

This module provides functions to categorize evaluation errors into different types
(e.g., syntax errors, wrong values, language mismatches, etc.).
"""

import json
from typing import Any, Dict
from config import ToolErrorCategory
from allow_synonym import _get_allow_synonym_backend


async def categorize_single_sample_async(evaluation_entry: Dict[str, Any]) -> ToolErrorCategory:
    """
    Asynchronously categorize a single evaluation sample to determine error type using LLM.

    Args:
        evaluation_entry: Evaluation result entry with 'id', 'valid', and error details

    Returns:
        ToolErrorCategory enum indicating the type of error
    """
    # If the sample is valid, it's not an error
    is_valid = evaluation_entry.get("valid", False)
    if is_valid:
        return ToolErrorCategory.EXACTLY_SAME_MEANING

    # Prepare the prompt for LLM categorization
    system_prompt = """You are an error categorization system. Given an evaluation error entry, determine which category the error belongs to.

CRITICAL: You must respond with EXACTLY one of these category strings (lowercase, no punctuation, no explanation):
- syntax_error
- misc_errors
- wrong_values
- language_mismatch
- relevant_but_incorrect
- exactly_same_meaning
- other_errors

Do not include any other text, explanation, or punctuation."""

    # Format the evaluation entry for the user prompt
    error_details = json.dumps(evaluation_entry, ensure_ascii=False, indent=2)

    user_prompt = f"""Evaluation error entry:
{error_details}

Based on the error details above, which category does this error belong to?
Respond with exactly one of: syntax_error, misc_errors, wrong_values, language_mismatch, relevant_but_incorrect, exactly_same_meaning, or other_errors."""

    # Get backend
    backend = _get_allow_synonym_backend()

    # Use the backend's client directly for chat completion
    from models.api_backend import APIBackend
    if isinstance(backend, APIBackend):
        client = backend.client
    else:
        raise TypeError(f"Expected APIBackend, got {type(backend)}")

    # Make API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = await client.chat.completions.create(
            model="gpt-5",  # Use the same model as allow_synonym
            messages=messages,
        )

        # Extract and validate response content
        if not response.choices or len(response.choices) == 0:
            print(f"Warning: LLM returned no choices for sample {evaluation_entry.get('id')}. Defaulting to other_errors.")
            return ToolErrorCategory.OTHER_ERRORS

        content = response.choices[0].message.content
        if content is None:
            print(f"Warning: LLM returned None content for sample {evaluation_entry.get('id')}. Defaulting to other_errors.")
            return ToolErrorCategory.OTHER_ERRORS

        raw_output = content.strip().lower()

        if not raw_output:
            print(f"Warning: LLM returned empty response for sample {evaluation_entry.get('id')}. Defaulting to other_errors.")
            return ToolErrorCategory.OTHER_ERRORS

        # Map the output to ToolErrorCategory enum
        category_map = {
            "syntax_error": ToolErrorCategory.SYNTAX_ERROR,
            "misc_errors": ToolErrorCategory.MISC_ERRORS,
            "wrong_values": ToolErrorCategory.WRONG_VALUES,
            "language_mismatch": ToolErrorCategory.LANGUAGE_MISMATCH,
            "relevant_but_incorrect": ToolErrorCategory.RELEVANT_BUT_INCORRECT,
            "exactly_same_meaning": ToolErrorCategory.EXACTLY_SAME_MEANING,
            "other_errors": ToolErrorCategory.OTHER_ERRORS,
        }

        if raw_output in category_map:
            return category_map[raw_output]
        else:
            print(f"Warning: LLM returned invalid category '{raw_output}' for sample {evaluation_entry.get('id')}. Defaulting to other_errors.")
            return ToolErrorCategory.OTHER_ERRORS

    except Exception as e:
        print(f"Error: Failed to categorize sample {evaluation_entry.get('id')}: {e}. Defaulting to other_errors.")
        return ToolErrorCategory.OTHER_ERRORS
