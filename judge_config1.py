"""
Sample configuration file for tool_main.py

This file demonstrates how to configure different combinations of:
- Models (API or Local)
- Translation modes
- Noise modes

Usage:
    python tool_main.py --config tool_config1.py --num-gpus 1
"""

from config import (
    JudgeConfig,
    ApiModel,
    LocalModel,
    Language,
    TranslateOption,
    AddNoiseMode,
    Translated,
    NotTranslated,
    ResultType,
)

# Example configurations
configs = [    
    JudgeConfig(
        model=LocalModel.QWEN3_8B,
        lang1="zh_cn",
        lang2="en",
        result_type=ResultType.PREFERENCE_DIRECT
    )
]

# Uncomment to generate all combinations programmatically
# configs = []
# for model in [ApiModel.GPT_5, ApiModel.GPT_5_MINI, ApiModel.GPT_5_NANO]:
#     for translate_mode in [
#         NotTranslated(),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME),
#         Translated(language=Language.CHINESE, option=TranslateOption.PARTIALLY_TRANSLATED),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE_POST_PROCESS_SAME),
#     ]:
#         for add_noise_mode in [
#             AddNoiseMode.NO_NOISE,
#             AddNoiseMode.SYNONYM,
#             AddNoiseMode.PARAPHRASE,
#         ]:
#             configs.append(
#                 ToolConfig(
#                     model=model,
#                     translate_mode=translate_mode,
#                     add_noise_mode=add_noise_mode,
#                 )
#             )
