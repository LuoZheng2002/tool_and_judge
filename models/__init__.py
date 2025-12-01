"""
Unified model interfaces and backends package.

This package provides a unified API for working with different LLM backends
(API, HuggingFace, vLLM) and model-specific interfaces across both the judge
and tool projects.

Key Components:
- ModelBackend: Abstract base class for inference backends
- ModelInterface: Abstract base class for model-specific behavior
- JudgeModelInterface: Extended interface for judge project (perplexity, preference)
- ToolModelInterface: Extended interface for tool project (function calling)
- create_backend: Factory function to create/cache backends
- create_interface: Factory function to create model interfaces

Usage Example:
    >>> from models import create_backend, create_interface
    >>>
    >>> # Create backend (cached automatically)
    >>> backend = create_backend(
    ...     backend_type="vllm",
    ...     model_name="Qwen/Qwen2.5-7B-Instruct",
    ...     num_gpus=1
    ... )
    >>>
    >>> # Create interface
    >>> interface = create_interface("Qwen/Qwen2.5-7B-Instruct")
    >>>
    >>> # Run inference (interface takes backend as argument)
    >>> result = interface.infer(
    ...     backend=backend,
    ...     user_query="What is the capital of France?",
    ...     max_new_tokens=50
    ... )
    >>> print(result)
"""

from .base import (
    ModelBackend,
    ModelInterface,
    JudgeModelInterface,
    ToolModelInterface,
    ForwardResult,
    GenerationResult,
    ComparisonResult,
)

from .model_factory import (
    create_backend,
    create_interface,
    shutdown_backend_cache,
    BackendConfig,
    BackendCache,
)

from .name_mapping import (
    FunctionNameMapper,
    get_global_name_mapper,
)

# Import model-specific interfaces
from .qwen3_interface import Qwen3Interface
from .gpt5_interface import GPT5Interface
from .deepseek_interface import DeepSeekInterface

__all__ = [
    # Base classes
    'ModelBackend',
    'ModelInterface',
    'JudgeModelInterface',
    'ToolModelInterface',

    # Result classes
    'ForwardResult',
    'GenerationResult',
    'ComparisonResult',

    # Factory functions
    'create_backend',
    'create_interface',
    'shutdown_backend_cache',

    # Configuration and cache
    'BackendConfig',
    'BackendCache',

    # Name mapping
    'FunctionNameMapper',
    'get_global_name_mapper',

    # Model-specific interfaces
    'Qwen3Interface',
    'GPT5Interface',
    'DeepSeekInterface',
]

__version__ = '0.1.0'
