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
    # Example: Test only GPT-5-NANO with no translation
    Config(
        model=ApiModel.GPT_5_NANO,
        translate_mode=NotTranslated(),
        add_noise_mode=AddNoiseMode.NO_NOISE
    ),

    # Example: Test GPT-5-MINI with Chinese translation
    # Config(
    #     model=ApiModel.GPT_5_MINI,
    #     translate_mode=Translated(
    #         language=Language.CHINESE,
    #         option=TranslateOption.FULLY_TRANSLATED
    #     ),
    #     add_noise_mode=AddNoiseMode.NO_NOISE
    # ),
]
