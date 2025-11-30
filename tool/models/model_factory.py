"""
Factory for creating model interface instances.

Provides a unified way to instantiate the correct model handler based on model type.
"""

from typing import Union, Optional, List, Dict, Any
from config import ApiModel, LocalModel
from models.base import ModelInterface
from models.gpt_4o_mini_interface import GPT4oMiniInterface
from models.gpt_5_interface import GPT5Interface
from models.claude_sonnet_interface import ClaudeSonnetInterface
from models.claude_haiku_interface import ClaudeHaikuInterface
from models.deepseek_chat_interface import DeepseekChatInterface
from models.llama_3_1_interface import Llama31Interface
from models.granite_3_1_8b_instruct_interface import Granite3_1_8BInstructInterface
from models.granite_4_interface import Granite4Interface
from models.qwen2_5_interface import Qwen25InstructInterface
from models.qwen3_interface import Qwen3Interface


def create_model_interface(model: Union[ApiModel, LocalModel]) -> ModelInterface:
    """
    Factory function to create appropriate model interface instance.

    Args:
        model: Either an ApiModel enum or LocalModel enum

    Returns:
        ModelInterface instance for the specified model

    Raises:
        ValueError: If model type is not supported
        EnvironmentError: If required API keys are missing
    """
    if isinstance(model, ApiModel):
        return _create_api_model_interface(model)
    elif isinstance(model, LocalModel):
        return _create_local_model_interface(model)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def _create_api_model_interface(model: ApiModel) -> ModelInterface:
    """
    Create interface for API-based models.

    Args:
        model: ApiModel enum value

    Returns:
        ModelInterface instance

    Raises:
        ValueError: If model is not supported
        EnvironmentError: If required API keys are missing
    """
    match model:
        case ApiModel.GPT_5:
            return GPT5Interface(model_variant="gpt-5", use_strict_mode=False)
        case ApiModel.GPT_5_MINI:
            return GPT5Interface(model_variant="gpt-5-mini", use_strict_mode=False)
        case ApiModel.GPT_5_NANO:
            return GPT5Interface(model_variant="gpt-5-nano", use_strict_mode=False)
        case ApiModel.DEEPSEEK_CHAT:
            return DeepseekChatInterface()
        case ApiModel.LLAMA_3_1_8B:
            return Llama31Interface(model_id="meta.llama3-1-8b-instruct-v1:0")
        case ApiModel.LLAMA_3_1_70B:
            return Llama31Interface(model_id="meta.llama3-1-70b-instruct-v1:0")
        case _:
            raise ValueError(f"Unsupported API model: {model}")


def _create_local_model_interface(model: LocalModel) -> ModelInterface:
    """
    Create interface for local models.

    Args:
        model: LocalModel enum value

    Returns:
        ModelInterface instance

    Raises:
        ValueError: If model is not supported
    """
    match model:
        # case LocalModel.GRANITE_3_1_8B_INSTRUCT:
        #     return Granite3_1_8BInstructInterface()
        case LocalModel.GRANITE_4_0_H_TINY:
            return Granite4Interface(model_id="ibm-granite/granite-4.0-h-tiny")
        case LocalModel.GRANITE_4_0_H_SMALL:
            return Granite4Interface(model_id="ibm-granite/granite-4.0-h-small")
        # case LocalModel.QWEN_2_5_7B_INSTRUCT:
        #     return Qwen25InstructInterface(model_id="Qwen/Qwen2.5-7B-Instruct")
        # case LocalModel.QWEN_2_5_14B_INSTRUCT:
        #     return Qwen25InstructInterface(model_id="Qwen/Qwen2.5-14B-Instruct")
        # case LocalModel.QWEN_2_5_32B_INSTRUCT:
        #     return Qwen25InstructInterface(model_id="Qwen/Qwen2.5-32B-Instruct")
        # case LocalModel.QWEN_2_5_72B_INSTRUCT:
        #     return Qwen25InstructInterface(model_id="Qwen/Qwen2.5-72B-Instruct")
        case LocalModel.QWEN3_8B:
            return Qwen3Interface(model_id="Qwen/Qwen3-8B")
        case LocalModel.QWEN3_14B:
            return Qwen3Interface(model_id="Qwen/Qwen3-14B")
        case LocalModel.QWEN3_32B:
            return Qwen3Interface(model_id="Qwen/Qwen3-32B-A3B")
        case LocalModel.QWEN3_NEXT_80B:
            return Qwen3Interface(model_id="Qwen/Qwen3-Next-80B-A3B-Instruct")
        case _:
            raise ValueError(f"Unsupported local model: {model}")
