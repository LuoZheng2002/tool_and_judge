"""
Sample configuration file for judge_run.py

This file demonstrates how to configure different combinations of:
- Judge models (local models for evaluation)
- Language pairs for comparison
- Result types (preference direct, preference CoT, or perplexity)

Usage:
    python judge_run.py --config judge_config1.py --num-gpus 1
"""

from config import (
    JudgeConfig,
    LocalModel,
    ResultType,
)

# Example configurations
configs = [
    # Example 1: Qwen 3 30B comparing Chinese vs English - Direct preference
    JudgeConfig(
        model=LocalModel.QWEN3_30B_A3B,
        lang1="zh_cn",
        lang2="en",
        result_type=ResultType.PREFERENCE_DIRECT
    ),

    # Example 2: Qwen 3 30B comparing Chinese vs English - Perplexity
    JudgeConfig(
        model=LocalModel.QWEN3_30B_A3B,
        lang1="zh_cn",
        lang2="en",
        result_type=ResultType.PERPLEXITY
    ),

    # Example 3: Qwen 3 30B comparing Chinese vs English - Chain of Thought preference
    JudgeConfig(
        model=LocalModel.QWEN3_30B_A3B,
        lang1="zh_cn",
        lang2="en",
        result_type=ResultType.PREFERENCE_COT
    ),

    # Example 4: Granite 3.1 8B comparing Hindi vs English - Direct preference
    JudgeConfig(
        model=LocalModel.GRANITE_3_1_8B_INSTRUCT,
        lang1="hi",
        lang2="en",
        result_type=ResultType.PREFERENCE_DIRECT
    ),

    # Example 5: Granite 3.1 8B comparing Hindi vs English - Perplexity
    JudgeConfig(
        model=LocalModel.GRANITE_3_1_8B_INSTRUCT,
        lang1="hi",
        lang2="en",
        result_type=ResultType.PERPLEXITY
    ),
]

# Uncomment to generate all combinations programmatically
# configs = []
# for model in [LocalModel.QWEN3_30B_A3B, LocalModel.GRANITE_3_1_8B_INSTRUCT]:
#     for lang_pair in [("zh_cn", "en"), ("hi", "en")]:
#         lang1, lang2 = lang_pair
#         for result_type in [ResultType.PREFERENCE_DIRECT, ResultType.PREFERENCE_COT, ResultType.PERPLEXITY]:
#             configs.append(
#                 JudgeConfig(
#                     model=model,
#                     lang1=lang1,
#                     lang2=lang2,
#                     result_type=result_type
#                 )
#             )
