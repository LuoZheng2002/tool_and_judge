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
    ToolConfig,
    ApiModel,
    LocalModel,
    Language,
    TranslateOption,
    AddNoiseMode,
    Translated,
    NotTranslated,
)

# Example configurations
configs = [    
    # ToolConfig(
    #     model=LocalModel.QWEN3_14B,
    #     translate_mode=Translated(
    #         language=Language.CHINESE,
    #         option=TranslateOption.FULLY_TRANSLATED
    #     ),
    #     # translate_mode=NotTranslated(),
    #     add_noise_mode=AddNoiseMode.NO_NOISE
    # ),
]

# Uncomment to generate all combinations programmatically
for model in [LocalModel.QWEN3_NEXT_80B]:
    for translate_mode in [
        NotTranslated(),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME),
        Translated(language=Language.CHINESE, option=TranslateOption.PARTIALLY_TRANSLATED),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE_POST_PROCESS_SAME),
    ]:
        for add_noise_mode in [
            AddNoiseMode.NO_NOISE,
            AddNoiseMode.SYNONYM,
            AddNoiseMode.PARAPHRASE,
        ]:
            configs.append(
                ToolConfig(
                    model=model,
                    translate_mode=translate_mode,
                    add_noise_mode=add_noise_mode,
                )
            )
