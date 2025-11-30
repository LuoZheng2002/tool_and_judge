from enum import Enum, auto
from dataclasses import dataclass

from typing import NamedTuple, Union

class ApiModel(Enum):
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    DEEPSEEK_CHAT = "deepseek-chat"
    LLAMA_3_1_8B = "meta.llama3-1-8b-instruct-v1:0"
    LLAMA_3_1_70B = "meta.llama3-1-70b-instruct-v1:0"

class LocalModel(Enum):
    GRANITE_4_0_H_TINY = "ibm-granite/granite-4.0-h-tiny"
    GRANITE_4_0_H_SMALL = "ibm-granite/granite-4.0-h-small"
    QWEN3_8B = "Qwen/Qwen3-8B"
    QWEN3_14B = "Qwen/Qwen3-14B"
    QWEN3_32B = "Qwen/Qwen3-32B-A3B"
    QWEN3_NEXT_80B = "Qwen/Qwen3-Next-80B-A3B-Instruct"

Model = Union[ApiModel, LocalModel]

# Models that require function name sanitization (e.g., GPT-5 doesn't allow dots in function names)
# This mapping is used to determine if we need to build name mappings for a model
MODEL_REQUIRES_NAME_SANITIZATION = {
    ApiModel.GPT_5: True,
    ApiModel.GPT_5_MINI: True,
    ApiModel.GPT_5_NANO: True,
    # All other models default to False (no sanitization needed)
}

def requires_name_sanitization(model: Model) -> bool:
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
    FULLY_TRANSLATED_POST_PROCESS_DIFFERENT = auto()
    FULLY_TRANSLATED_POST_PROCESS_SAME = auto(),
    FULLY_TRANSLATED_PROMPT_TRANSLATE_POST_PROCESS_SAME = auto(),

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

class PostProcessOption(Enum):
    DONT_POST_PROCESS = 0
    POST_PROCESS_DIFFERENT = 1
    POST_PROCESS_SAME = 2

@dataclass(frozen=True)
class Config:
    model: Model
    translate_mode: TranslateMode
    add_noise_mode: AddNoiseMode

# configs: list[Config] = [

#     ]

# configs.append(Config(model=ApiModel.GPT_5, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE))
# for model in []:
#     for translate_mode in [
#         NotTranslated(),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE_POST_PROCESS_SAME),
#         Translated(language=Language.CHINESE, option=TranslateOption.PARTIALLY_TRANSLATED),
#     ]:
#         for add_noise_mode in [
#             AddNoiseMode.NO_NOISE,
#             AddNoiseMode.SYNONYM,
#             AddNoiseMode.PARAPHRASE,
#         ]:
#             configs.append(
#                 Config(
#                     model=model,
#                     translate_mode=translate_mode,
#                     add_noise_mode=add_noise_mode,
#                 )
#             )


requires_inference_raw = True
requires_inference_json = True
requires_post_processing = True # rephrase parameter values if the raw output has a similar meaning as the ground truth but is not an exact match
requires_evaluation = True
requires_score = True

evaluation_caching = False


