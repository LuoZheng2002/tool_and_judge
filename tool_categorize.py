"""
Categorization module for analyzing and grouping evaluation errors by type.

This module provides functions to categorize evaluation errors into different types
(e.g., syntax errors, wrong values, language mismatches, etc.).
"""

import json
import re
from typing import Any, Dict, Tuple, List
from config import ToolErrorCategory, EvaluationError
from allow_synonym import _get_allow_synonym_backend


async def _categorize_parameter_value_async(
    param_name: str,
    actual_value: Any,
    expected_values: List[Any]
) -> Tuple[ToolErrorCategory, str]:
    """
    Use LLM to categorize a single parameter value mismatch into one of 6 categories.

    Args:
        param_name: Name of the parameter
        actual_value: The actual value from model output
        expected_values: List of expected values from ground truth

    Returns:
        Tuple of (ToolErrorCategory enum, raw_llm_response string)
    """
    system_prompt = """You are a parameter value categorization system. Given a parameter with its actual value and expected values, determine which category the mismatch belongs to.

Here are the 6 available categories for parameter value mismatches:
1. wrong_values: The output value is COMPLETELY incorrect (wrong calculation, wrong fact, unrelated content). If some words or meanings overlap with expected values, choose relevant_but_incorrect instead.
2. relevant_but_incorrect: The value is in English, relevant to the expected values, but not exactly the same in meaning.
3. exactly_same_meaning: The value is in English and conveys the exact same meaning as one of the expected values, though not verbatim.
4. language_mismatch_wrong_values: The value contains non-English text AND is completely incorrect.
5. language_mismatch_relevant_but_incorrect: The value contains non-English text AND is relevant but not exactly correct.
6. language_mismatch_exactly_same_meaning: The value contains non-English text AND conveys the same meaning as expected.

CRITICAL: You must put your final decision inside \\boxed{} like this: \\boxed{category_name}
where category_name is exactly one of: wrong_values, relevant_but_incorrect, exactly_same_meaning, language_mismatch_wrong_values, language_mismatch_relevant_but_incorrect, or language_mismatch_exactly_same_meaning."""

    user_prompt = f"""Parameter: {param_name}
Actual value: {json.dumps(actual_value, ensure_ascii=False)}
Expected values: {json.dumps(expected_values, ensure_ascii=False)}

Which category does this parameter value mismatch belong to?
Put your final answer in \\boxed{{category_name}}."""

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
            model="gpt-5",
            messages=messages,
        )

        # Extract and validate response content
        if not response.choices or len(response.choices) == 0:
            return ToolErrorCategory.OTHER_ERRORS, "Error: No choices returned"

        content = response.choices[0].message.content
        if content is None or not content.strip():
            return ToolErrorCategory.OTHER_ERRORS, "Error: Empty response"

        raw_response = content

        # Extract category from \boxed{category_name}
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, content)

        if not match:
            return ToolErrorCategory.OTHER_ERRORS, raw_response

        raw_category = match.group(1).strip().lower()

        # Map to ToolErrorCategory enum
        category_map = {
            "wrong_values": ToolErrorCategory.WRONG_VALUES,
            "relevant_but_incorrect": ToolErrorCategory.RELEVANT_BUT_INCORRECT,
            "exactly_same_meaning": ToolErrorCategory.EXACTLY_SAME_MEANING,
            "language_mismatch_wrong_values": ToolErrorCategory.LANGUAGE_MISMATCH_WRONG_VALUES,
            "language_mismatch_relevant_but_incorrect": ToolErrorCategory.LANGUAGE_MISMATCH_RELEVANT_BUT_INCORRECT,
            "language_mismatch_exactly_same_meaning": ToolErrorCategory.LANGUAGE_MISMATCH_EXACTLY_SAME_MEANING,
        }

        if raw_category in category_map:
            return category_map[raw_category], raw_response
        else:
            return ToolErrorCategory.OTHER_ERRORS, raw_response

    except Exception as e:
        return ToolErrorCategory.OTHER_ERRORS, f"Error: {str(e)}"


async def categorize_single_sample_async(evaluation_entry: Dict[str, Any]) -> Tuple[ToolErrorCategory, str]:
    """
    Asynchronously categorize a single evaluation sample to determine error type.

    Filters out errors that don't need LLM:
    - Syntax errors (from postprocess): Mapped to SYNTAX_ERROR directly
    - Misc errors (from evaluation): Mapped to MISC_ERRORS directly

    Only calls LLM for INVALID_PARAM_VALUE errors, using per-parameter categorization.

    Args:
        evaluation_entry: Evaluation result entry with 'id', 'valid', 'error', and 'error_meta'

    Returns:
        Tuple of (ToolErrorCategory enum, raw_llm_response string)
    """
    # If the sample is valid, it shouldn't be categorized
    is_valid = evaluation_entry.get("valid", False)
    assert not is_valid, "Expected invalid sample for categorization"

    # Get the error type
    error_type = evaluation_entry.get("error", "")

    # Map postprocess errors to categories directly (no LLM needed)
    if error_type == EvaluationError.NO_FUNCTION_CALLS_FOUND.value:
        return ToolErrorCategory.SYNTAX_ERROR, "Postprocess error: no function calls found"
    elif error_type == EvaluationError.JSON_DECODE_ERROR.value:
        return ToolErrorCategory.SYNTAX_ERROR, "Postprocess error: JSON decode error"
    elif error_type == EvaluationError.PARSING_ERROR.value:
        return ToolErrorCategory.SYNTAX_ERROR, "Postprocess error: parsing error"

    # Map evaluation errors to categories
    elif error_type == EvaluationError.INVALID_ENTRY_COUNT.value:
        return ToolErrorCategory.MISC_ERRORS, "Evaluation error: invalid entry count"
    elif error_type == EvaluationError.WRONG_FUNC_NAME.value:
        return ToolErrorCategory.MISC_ERRORS, "Evaluation error: wrong function name"
    elif error_type == EvaluationError.MISSING_REQUIRED_PARAM.value:
        return ToolErrorCategory.MISC_ERRORS, "Evaluation error: missing required parameter"
    elif error_type == EvaluationError.UNEXPECTED_PARAM.value:
        return ToolErrorCategory.MISC_ERRORS, "Evaluation error: unexpected parameter"

    # For INVALID_PARAM_VALUE, use LLM to categorize the specific parameter
    elif error_type == EvaluationError.INVALID_PARAM_VALUE.value:
        error_meta = evaluation_entry.get("error_meta", {})
        param = error_meta.get("param", "unknown")
        actual_value = error_meta.get("actual_value")
        expected_values = error_meta.get("expected_values", [])

        # Call LLM to categorize this specific parameter value
        category, llm_response = await _categorize_parameter_value_async(
            param, actual_value, expected_values
        )
        return category, llm_response

    else:
        # Unknown error type
        return ToolErrorCategory.OTHER_ERRORS, f"Unknown error type: {error_type}"
