"""
Example custom configuration file.

This file only needs to define the 'configs' list.
All other parameters (requires_*, evaluation_caching) are read from config.py.

Usage:
    python main.py --config config_example.py
"""

from config import (
    Config, ApiModel, LocalModel,
    Language, TranslateOption, AddNoiseMode,
    Translated, NotTranslated
)

# Define your custom configs list here
configs = [
]

for model in [LocalModel.QWEN3_8B]:
    for translate_mode in [
        NotTranslated(),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE_POST_PROCESS_SAME),
        Translated(language=Language.CHINESE, option=TranslateOption.PARTIALLY_TRANSLATED),
    ]:
        for add_noise_mode in [
            AddNoiseMode.NO_NOISE,
            AddNoiseMode.SYNONYM,
            AddNoiseMode.PARAPHRASE,
        ]:
            configs.append(
                Config(
                    model=model,
                    translate_mode=translate_mode,
                    add_noise_mode=add_noise_mode,
                )
            )

