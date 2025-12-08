from enum import Enum, auto
from dataclasses import dataclass

from typing import NamedTuple, Union

# ============================================================================
# Tool Project Configuration
# ============================================================================

class ApiModel(Enum):
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_4O_MINI = "gpt-4o-mini"  # Used for post-processing
    DEEPSEEK_CHAT = "deepseek-chat"
    LLAMA_3_1_8B = "meta.llama3-1-8b-instruct-v1:0"
    LLAMA_3_1_70B = "meta.llama3-1-70b-instruct-v1:0"

class LocalModel(Enum):
    # Granite models
    GRANITE_3_1_8B_INSTRUCT = "ibm-granite/granite-3.1-8b-instruct"
    GRANITE_4_0_H_TINY = "ibm-granite/granite-4.0-h-tiny"
    GRANITE_4_0_H_SMALL = "ibm-granite/granite-4.0-h-small"
    # Qwen models
    QWEN3_8B = "Qwen/Qwen3-8B"
    QWEN3_14B = "Qwen/Qwen3-14B"
    QWEN3_30B_A3B = "Qwen/Qwen3-30B-A3B"
    QWEN3_32B = "Qwen/Qwen3-32B-A3B"
    QWEN3_NEXT_80B_A3B = "Qwen/Qwen3-Next-80B-A3B-Instruct"

ToolModel = Union[ApiModel, LocalModel]

# Models that require function name sanitization (e.g., GPT-5 doesn't allow dots in function names)
# This mapping is used to determine if we need to build name mappings for a model
MODEL_REQUIRES_NAME_SANITIZATION = {
    ApiModel.GPT_5: True,
    ApiModel.GPT_5_MINI: True,
    ApiModel.GPT_5_NANO: True,
    # All other models default to False (no sanitization needed)
}

def requires_name_sanitization(model: ToolModel) -> bool:
    """
    Check if a model requires function name sanitization.

    Args:
        model: ApiModel or LocalModel enum

    Returns:
        True if the model requires name sanitization, False otherwise
    """
    return MODEL_REQUIRES_NAME_SANITIZATION.get(model, False)

class Language(Enum):
    CHINESE = auto()
    HINDI = auto()

class TranslateOption(Enum):
    FULLY_TRANSLATED = auto()
    FULLY_TRANSLATED_PROMPT_TRANSLATE = auto()
    PARTIALLY_TRANSLATED = auto()
    FULLY_TRANSLATED_PRE_TRANSLATE = auto()
    FULLY_TRANSLATED_POST_TRANSLATE = auto()

class AddNoiseMode(Enum):
    NO_NOISE = auto()
    SYNONYM = auto()
    PARAPHRASE = auto()

@dataclass(frozen=True)
class Translated:
    language: Language
    option: TranslateOption

@dataclass(frozen=True)
class NotTranslated:
    pass

TranslateMode = Union[Translated, NotTranslated]

@dataclass(frozen=True)
class ToolConfig:
    model: ToolModel
    translate_mode: TranslateMode
    add_noise_mode: AddNoiseMode

# Tool processing configuration
evaluation_caching = False

class ToolErrorCategory(Enum):
    SYNTAX_ERROR = "syntax_error"
    MISC_ERRORS = "misc_errors"
    WRONG_VALUES = "wrong_values"
    LANGUAGE_MISMATCH = "language_mismatch"
    RELEVANT_BUT_INCORRECT = "relevant_but_incorrect"
    EXACTLY_SAME_MEANING = "exactly_same_meaning"
    OTHER_ERRORS = "other_errors"

class EvaluationError(Enum):
    """Error types that can occur during postprocess and evaluation."""
    # Postprocess errors (from postprocess_tool_calls)
    NO_FUNCTION_CALLS_FOUND = "no_function_calls_found"  # Response parsed but no function calls extracted
    JSON_DECODE_ERROR = "json_decode_error"  # Failed to parse JSON output
    PARSING_ERROR = "parsing_error"  # General error during parsing

    # Evaluation errors (from evaluate_json)
    DECODER_FAILED = "decoder_failed"  # Decoded output is a string (decoder failed)
    INVALID_ENTRY_COUNT = "invalid_entry_count"  # Expected exactly one AST entry
    WRONG_FUNC_NAME = "wrong_func_name"  # Function name mismatch
    MISSING_REQUIRED_PARAM = "missing_required_param"  # Missing required parameter
    UNEXPECTED_PARAM = "unexpected_param"  # Unexpected parameter
    INVALID_PARAM_VALUE = "invalid_param_value"  # Invalid parameter value


# ============================================================================
# Judge Project Configuration
# ============================================================================

class ResultType(Enum):
    PREFERENCE_DIRECT = auto()
    PREFERENCE_COT = auto()
    PERPLEXITY = auto()

@dataclass(frozen=True)
class JudgeConfig:
    model: LocalModel
    lang1: str
    lang2: str
    result_type: ResultType


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# For backward compatibility with tool project imports
Model = ToolModel  # tool/config.py used "Model" as Union[ApiModel, LocalModel]
Config = ToolConfig  # tool/config.py used "Config" for tool configuration
