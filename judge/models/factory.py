"""
Factory functions to create the appropriate model interface and backend based on model name.
"""

import re
from typing import Any, Optional
from .base import ModelInterface, AsyncModelBackend
from .qwen3 import Qwen3ModelInterface
from .granite import GraniteModelInterface
from .hf_backend import HuggingFaceBackend
from .vllm_backend import VLLMBackend


def extract_model_size_in_billions(model_name: str) -> Optional[float]:
    """
    Extract the model size in billions of parameters from the model name.

    Args:
        model_name: HuggingFace model name (e.g., "ibm-granite/granite-3.1-8b-instruct", "Qwen/Qwen3-30B-A3B")

    Returns:
        Model size in billions as a float, or None if size cannot be extracted
    """
    # Look for patterns like "8b", "30B", "7b-instruct", etc.
    # Case-insensitive search for number followed by 'b'
    pattern = r'(\d+\.?\d*)[bB]'
    match = re.search(pattern, model_name)

    if match:
        size_str = match.group(1)
        return float(size_str)

    return None


def create_model_interface(model_name: str) -> ModelInterface:
    """
    Create the appropriate model interface based on the model name.

    This function examines the model name and returns the corresponding
    ModelInterface implementation. It supports:
    - Qwen models (Qwen/Qwen2.5-*)
    - Granite models (ibm-granite/granite-*)

    Args:
        model_name: Hugging Face model name (e.g., "Qwen/Qwen2.5-7B-Instruct")

    Returns:
        ModelInterface instance appropriate for the given model

    Raises:
        ValueError: If the model name is not recognized

    Examples:
        >>> interface = create_model_interface("Qwen/Qwen2.5-7B-Instruct")
        >>> isinstance(interface, Qwen3ModelInterface)
        True

        >>> interface = create_model_interface("ibm-granite/granite-3.1-8b-instruct")
        >>> isinstance(interface, GraniteModelInterface)
        True
    """
    model_name_lower = model_name.lower()

    # Check for Qwen models
    if "qwen" in model_name_lower:
        return Qwen3ModelInterface()

    # Check for Granite models
    if "granite" in model_name_lower:
        return GraniteModelInterface()

    # If no match found, raise an error with helpful message
    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Supported models are: Qwen (Qwen/*), Granite (ibm-granite/granite-*). "
        f"To add support for this model, create a new ModelInterface implementation "
        f"in the models/ directory and update the factory function."
    )


def create_model_backend(
    backend_type: str,
    model: Any = None,
    tokenizer: Any = None,
    model_name: Optional[str] = None,
    device: str = "cuda",
    max_batch_size: int = 8,
    max_batch_wait: float = 0.05,
    num_gpus: int = 1,
    **kwargs
) -> AsyncModelBackend:
    """
    Create the appropriate async model backend.

    Args:
        backend_type: Type of backend ("huggingface" or "vllm")
        model: Model instance (required for huggingface backend)
        tokenizer: Tokenizer instance (required for both backends)
        model_name: Model name (required for vllm backend, optional for huggingface)
        device: Device to use (for huggingface backend)
        max_batch_size: Maximum batch size (default for huggingface backend if not auto-calculated)
        max_batch_wait: Maximum wait time for batching (for huggingface backend)
        num_gpus: Number of GPUs to use (for batch size calculation in huggingface backend)
        **kwargs: Additional backend-specific arguments

    Returns:
        AsyncModelBackend instance

    Raises:
        ValueError: If backend_type is not recognized or required args are missing

    Examples:
        >>> # Create HuggingFace backend
        >>> backend = create_model_backend(
        ...     "huggingface",
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     device="cuda",
        ...     num_gpus=2
        ... )

        >>> # Create vLLM backend
        >>> backend = create_model_backend(
        ...     "vllm",
        ...     model_name="Qwen/Qwen2.5-7B-Instruct",
        ...     tokenizer=tokenizer
        ... )
    """
    backend_type_lower = backend_type.lower()

    if backend_type_lower == "huggingface" or backend_type_lower == "hf":
        if model is None or tokenizer is None:
            raise ValueError("HuggingFace backend requires 'model' and 'tokenizer' arguments")

        # Calculate batch size based on model size and num_gpus for HuggingFace backend
        calculated_batch_size = max_batch_size  # default
        if model_name is not None:
            model_size_in_billions = extract_model_size_in_billions(model_name)
            if model_size_in_billions is not None:
                # Formula: batch_size * model_size_in_billions = 120 * num_gpus
                calculated_batch_size = int(120 * num_gpus / model_size_in_billions)
                print(f"Model size: {model_size_in_billions}B parameters")
                print(f"Calculated batch size: {calculated_batch_size} (for {num_gpus} GPU(s))")
            else:
                print(f"Warning: Could not extract model size from {model_name}, using default batch size {max_batch_size}")
        else:
            print(f"Warning: model_name not provided, using default batch size {max_batch_size}")

        return HuggingFaceBackend(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_batch_size=calculated_batch_size,
            max_batch_wait=max_batch_wait
        )

    elif backend_type_lower == "vllm":
        if model_name is None or tokenizer is None:
            raise ValueError("vLLM backend requires 'model_name' and 'tokenizer' arguments")

        # Extract vLLM-specific kwargs
        tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.9)
        # max_model_len = kwargs.get('max_model_len', None)

        return VLLMBackend(
            model_name=model_name,
            tokenizer=tokenizer,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=1000 # Zheng: modify here for max_model_len
        )

    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Supported backends are: 'huggingface' (or 'hf'), 'vllm'."
        )
