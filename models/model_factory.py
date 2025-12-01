"""
Factory functions and caching system for model backends and interfaces.

This module provides:
- BackendCache: Manages backend lifecycle and caching
- create_backend: Creates/retrieves cached backends (API, HuggingFace, vLLM)
- create_interface: Creates model-specific interfaces

The caching system ensures backends are reused across different configurations,
reducing memory usage and improving performance.
"""

import os
import re
import asyncio
from typing import Optional, Any, Dict, Union
from dataclasses import dataclass

from .base import ModelBackend, ModelInterface

from models.vllm_backend import VLLMBackend


# =============================================================================
# Backend Configuration Data Classes
# =============================================================================

@dataclass
class BackendConfig:
    """Configuration for a model backend."""

    backend_type: str  # "api", "huggingface", "vllm"
    model_name: str  # Model identifier/name

    # Optional parameters
    api_key: Optional[str] = None  # For API backends
    base_url: Optional[str] = None  # For custom API endpoints
    device: str = "cuda"  # For local backends
    num_gpus: int = 1  # For local backends
    
    gpu_memory_utilization: float = 0.9  # For vLLM
    max_model_len: Optional[int] = None  # For vLLM
    max_batch_size: Optional[int] = None  # For HuggingFace
    max_batch_wait: float = 0.05  # For HuggingFace

    def get_cache_key(self) -> str:
        """
        Generate a unique cache key for this backend configuration.

        Returns:
            String key that uniquely identifies this backend configuration
        """
        return f"{self.backend_type}:{self.model_name}:{self.num_gpus}"


# =============================================================================
# Backend Cache
# =============================================================================

class BackendCache:
    """
    Manages backend lifecycle and caching.

    This class ensures that:
    - Multiple named backends can be cached (e.g., "experiment", "allow_synonym")
    - Backends are properly cleaned up when switching
    - Backends are reused when the same configuration is requested
    """

    def __init__(self):
        # Map from instance_name to (backend, config) tuple
        self._backends: Dict[str, tuple[Optional[ModelBackend], Optional[BackendConfig]]] = {}

    def get_or_create(
        self,
        config: BackendConfig,
        creator_func: callable,
        instance_name: str = "default"
    ) -> ModelBackend:
        """
        Get cached backend or create a new one.

        If the requested configuration matches the cached backend for this instance, return it.
        Otherwise, shut down the old backend for this instance and create a new one.

        Args:
            config: Backend configuration
            creator_func: Function that creates the backend (called if cache miss)
            instance_name: Name of the backend instance (e.g., "experiment", "allow_synonym")

        Returns:
            ModelBackend instance
        """
        cache_key = config.get_cache_key()

        # Check if we have a cached backend for this instance
        if instance_name in self._backends:
            current_backend, current_config = self._backends[instance_name]

            if current_backend is not None and current_config is not None:
                current_key = current_config.get_cache_key()

                if current_key == cache_key:
                    # For vLLM backends, check if the event loop has changed
                    if config.backend_type == "vllm":
                        try:
                            import asyncio
                            # Try to get the current running loop
                            try:
                                current_loop = asyncio.get_running_loop()
                                # Check if backend's engine is still alive in this loop
                                # If we're in a different loop, recreate the backend
                                loop_key = f"{instance_name}_loop_id"
                                if not hasattr(self, loop_key) or getattr(self, loop_key) != id(current_loop):
                                    print(f"Event loop changed, recreating vLLM backend [{instance_name}]: {cache_key}")
                                    self._cleanup_backend(instance_name)
                                    new_backend = creator_func(config)
                                    self._backends[instance_name] = (new_backend, config)
                                    setattr(self, loop_key, id(current_loop))
                                    return new_backend
                            except RuntimeError:
                                # No running loop - backend was created outside async context
                                # Force recreation if we had a previous loop
                                loop_key = f"{instance_name}_loop_id"
                                if hasattr(self, loop_key):
                                    print(f"No active event loop, recreating vLLM backend [{instance_name}]: {cache_key}")
                                    self._cleanup_backend(instance_name)
                                    new_backend = creator_func(config)
                                    self._backends[instance_name] = (new_backend, config)
                                    delattr(self, loop_key)
                                    return new_backend
                        except ImportError:
                            pass

                    print(f"Reusing cached backend [{instance_name}]: {cache_key}")
                    return current_backend

                # Different backend needed - cleanup current one for this instance
                print(f"Switching backend [{instance_name}] from {current_key} to {cache_key}")
                self._cleanup_backend(instance_name)

        # Create new backend
        print(f"Creating new backend [{instance_name}]: {cache_key}")
        new_backend = creator_func(config)
        self._backends[instance_name] = (new_backend, config)

        # Track event loop for vLLM backends
        if config.backend_type == "vllm":
            try:
                import asyncio
                try:
                    current_loop = asyncio.get_running_loop()
                    loop_key = f"{instance_name}_loop_id"
                    setattr(self, loop_key, id(current_loop))
                except RuntimeError:
                    # No running loop yet
                    loop_key = f"{instance_name}_loop_id"
                    if hasattr(self, loop_key):
                        delattr(self, loop_key)
            except ImportError:
                pass

        return new_backend

    def _cleanup_backend(self, instance_name: str):
        """Cleanup a specific backend instance and free resources."""
        if instance_name not in self._backends:
            return

        backend, config = self._backends[instance_name]
        if backend is None:
            return

        try:
            # Call async shutdown
            asyncio.run(backend.shutdown())
        except Exception as e:
            print(f"Warning: Error during backend cleanup [{instance_name}]: {e}")

        # Remove from cache
        del self._backends[instance_name]

        # Force garbage collection for local model backends
        import gc
        gc.collect()
        gc.collect()

        # Try to free CUDA memory if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"CUDA memory freed [{instance_name}]")
        except ImportError:
            pass

    def shutdown(self, instance_name: Optional[str] = None):
        """
        Shutdown and cleanup backends.

        Args:
            instance_name: Name of specific instance to shutdown. If None, shutdown all backends.
        """
        if instance_name is not None:
            self._cleanup_backend(instance_name)
        else:
            # Shutdown all backends
            for name in list(self._backends.keys()):
                self._cleanup_backend(name)


# Global backend cache instance
_global_backend_cache = BackendCache()


# =============================================================================
# Backend Creation Functions
# =============================================================================

def _extract_model_size_in_billions(model_name: str) -> Optional[float]:
    """
    Extract the model size in billions of parameters from the model name.

    Args:
        model_name: Model name (e.g., "Qwen/Qwen2.5-7B-Instruct", "granite-3.1-8b")

    Returns:
        Model size in billions as a float, or None if size cannot be extracted
    """
    # Look for patterns like "8b", "30B", "7b-instruct", etc.
    pattern = r'(\d+\.?\d*)[bB]'
    match = re.search(pattern, model_name)

    if match:
        size_str = match.group(1)
        return float(size_str)

    return None


def _create_api_backend(config: BackendConfig) -> ModelBackend:
    """
    Create an API-based backend.

    Args:
        config: Backend configuration

    Returns:
        ModelBackend instance for API inference

    Raises:
        ImportError: If required API backend module is not available
        ValueError: If API configuration is invalid
    """
    # Import the API backend implementation
    try:
        from .api_backend import APIBackend
    except ImportError:
        raise ImportError(
            "API backend not available. Please ensure models/api_backend.py exists "
            "implementing the APIBackend class."
        )

    return APIBackend(
        model_name=config.model_name,
        api_key=config.api_key,
        base_url=config.base_url
    )


def _create_huggingface_backend(config: BackendConfig) -> ModelBackend:
    """
    Create a HuggingFace transformers backend.

    Args:
        config: Backend configuration

    Returns:
        ModelBackend instance for HuggingFace inference

    Raises:
        ImportError: If transformers or torch not installed
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError(
            "HuggingFace backend requires transformers and torch. "
            "Install with: pip install transformers torch"
        )

    # Import the HuggingFace backend implementation
    try:
        from .hf_backend import HuggingFaceBackend
    except ImportError:
        raise ImportError(
            "HuggingFace backend not available. Please create models/hf_backend.py "
            "implementing the HuggingFaceBackend class."
        )

    print(f"Loading model: {config.model_name}")
    print(f"Using device: {config.device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Calculate batch size if not provided
    if config.max_batch_size is None:
        model_size = _extract_model_size_in_billions(config.model_name)
        if model_size is not None:
            # Formula: batch_size * model_size = 120 * num_gpus
            calculated_batch_size = int(120 * config.num_gpus / model_size)
            print(f"Model size: {model_size}B parameters")
            print(f"Calculated batch size: {calculated_batch_size} (for {config.num_gpus} GPU(s))")
            max_batch_size = calculated_batch_size
        else:
            max_batch_size = 8  # default
            print(f"Warning: Could not extract model size, using default batch size {max_batch_size}")
    else:
        max_batch_size = config.max_batch_size

    return HuggingFaceBackend(
        model=model,
        tokenizer=tokenizer,
        device=config.device,
        max_batch_size=max_batch_size,
        max_batch_wait=config.max_batch_wait
    )


def _create_vllm_backend(config: BackendConfig) -> ModelBackend:
    """
    Create a vLLM backend.

    Args:
        config: Backend configuration

    Returns:
        ModelBackend instance for vLLM inference

    Raises:
        ImportError: If vllm not installed
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "vLLM backend requires transformers. "
            "Install with: pip install transformers"
        )


    print(f"Loading tokenizer for: {config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )

    # Determine tensor parallel size
    tensor_parallel_size = config.num_gpus

    return VLLMBackend(
        model_name=config.model_name,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len
    )


# =============================================================================
# Public Factory Functions
# =============================================================================

def create_backend(
    backend_type: str,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    device: str = "cuda",
    num_gpus: int = 1,
    use_cache: bool = True,
    instance_name: str = "default",
    **kwargs
) -> ModelBackend:
    """
    Create or retrieve a cached model backend.

    This is the main entry point for creating backends. It handles:
    - Backend creation based on type (api, huggingface, vllm)
    - Caching and reuse of backends
    - Proper cleanup when switching backends

    Args:
        backend_type: Type of backend ("api", "huggingface", "vllm")
        model_name: Model identifier/name
        api_key: API key (for API backends)
        base_url: Custom API endpoint (for API backends)
        device: Device to use (for local backends)
        num_gpus: Number of GPUs to use (for local backends)
        use_cache: Whether to use backend caching (default: True)
        instance_name: Name for this backend instance (default: "default")
                      Use different names for different purposes (e.g., "experiment", "allow_synonym")
        **kwargs: Additional backend-specific parameters

    Returns:
        ModelBackend instance

    Raises:
        ValueError: If backend_type is not supported
        ImportError: If required dependencies are not installed

    Examples:
        >>> # Create API backend for experiments
        >>> backend = create_backend(
        ...     backend_type="api",
        ...     model_name="gpt-4",
        ...     api_key="sk-...",
        ...     instance_name="experiment"
        ... )

        >>> # Create separate API backend for allow_synonym
        >>> backend = create_backend(
        ...     backend_type="api",
        ...     model_name="gpt-5",
        ...     api_key="sk-...",
        ...     instance_name="allow_synonym"
        ... )

        >>> # Create HuggingFace backend
        >>> backend = create_backend(
        ...     backend_type="huggingface",
        ...     model_name="Qwen/Qwen2.5-7B-Instruct",
        ...     num_gpus=2
        ... )

        >>> # Create vLLM backend
        >>> backend = create_backend(
        ...     backend_type="vllm",
        ...     model_name="Qwen/Qwen2.5-7B-Instruct",
        ...     num_gpus=1
        ... )
    """

    print("Creating backend...")
    backend_type = backend_type.lower()

    # Create configuration
    config = BackendConfig(
        backend_type=backend_type,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        device=device,
        num_gpus=num_gpus,
        gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9),
        max_model_len=kwargs.get('max_model_len'),
        max_batch_size=kwargs.get('max_batch_size'),
        max_batch_wait=kwargs.get('max_batch_wait', 0.05)
    )

    # Define creator function based on backend type
    if backend_type == "api":
        creator_func = _create_api_backend
    elif backend_type in ["huggingface", "hf"]:
        creator_func = _create_huggingface_backend
    elif backend_type == "vllm":
        creator_func = _create_vllm_backend
    else:
        raise ValueError(
            f"Unsupported backend type: {backend_type}. "
            f"Supported types are: 'api', 'huggingface' (or 'hf'), 'vllm'."
        )

    # Use cache or create directly
    if use_cache:
        return _global_backend_cache.get_or_create(config, creator_func, instance_name=instance_name)
    else:
        return creator_func(config)


def create_interface(
    model_identifier: Union[str, Any],
    interface_type: str = "auto"
) -> ModelInterface:
    """
    Create a model-specific interface.

    This function creates the appropriate ModelInterface implementation based on
    the model identifier. It supports:
    - Auto-detection based on model name
    - Explicit interface type specification

    Args:
        model_identifier: Model name string or enum value
        interface_type: Type of interface ("auto", "judge", "tool", or specific model family)

    Returns:
        ModelInterface instance

    Raises:
        ValueError: If model is not recognized or interface type is invalid

    Examples:
        >>> # Auto-detect interface from model name
        >>> interface = create_interface("Qwen/Qwen2.5-7B-Instruct")

        >>> # Explicit interface type
        >>> interface = create_interface("ibm-granite/granite-3.1-8b", interface_type="granite")
    """
    # Convert enum to string if needed
    if hasattr(model_identifier, 'value'):
        model_name = model_identifier.value
    else:
        model_name = str(model_identifier)

    model_name_lower = model_name.lower()

    # Auto-detect or use explicit interface type
    if interface_type == "auto":
        # Detect based on model name
        if "qwen" in model_name_lower:
            detected_type = "qwen"
        elif "granite" in model_name_lower:
            detected_type = "granite"
        elif "gpt" in model_name_lower:
            detected_type = "gpt"
        elif "claude" in model_name_lower:
            detected_type = "claude"
        elif "llama" in model_name_lower:
            detected_type = "llama"
        elif "deepseek" in model_name_lower:
            detected_type = "deepseek"
        else:
            raise ValueError(
                f"Could not auto-detect interface type for model: {model_name}. "
                f"Please specify interface_type explicitly."
            )
    else:
        detected_type = interface_type.lower()

    # Import and create appropriate interface
    # Note: This assumes you have interface implementations for each model family
    # You'll need to create these based on the existing implementations in judge/tool folders

    if detected_type == "qwen":
        try:
            from .qwen3_interface import Qwen3Interface
            # Determine enable_thinking based on model name
            enable_thinking = False  # Default: disable thinking mode
            return Qwen3Interface(model_name=model_name, enable_thinking=enable_thinking)
        except ImportError:
            raise ImportError(
                f"Qwen interface not available. Please ensure models/qwen3_interface.py exists."
            )

    elif detected_type == "gpt":
        try:
            from .gpt5_interface import GPT5Interface
            # Determine model variant
            if "gpt-5-mini" in model_name.lower():
                variant = "gpt-5-mini"
            elif "gpt-5-nano" in model_name.lower():
                variant = "gpt-5-nano"
            elif "gpt-5" in model_name.lower():
                variant = "gpt-5"
            else:
                # Default to gpt-5 for other GPT models
                variant = "gpt-5"

            # Use non-strict mode by default for better compatibility
            return GPT5Interface(model_variant=variant, use_strict_mode=False)
        except ImportError:
            raise ImportError(
                f"GPT-5 interface not available. Please ensure models/gpt5_interface.py exists."
            )

    elif detected_type == "granite":
        try:
            from .granite_interface import GraniteInterface
            return GraniteInterface(model_name=model_name)
        except ImportError:
            raise ImportError(
                f"Granite interface not available. Please create models/granite_interface.py "
                f"or copy from existing judge/tool implementations."
            )

    elif detected_type == "claude":
        try:
            from .claude_interface import ClaudeInterface
            return ClaudeInterface(model_name=model_name)
        except ImportError:
            raise ImportError(
                f"Claude interface not available. Please create models/claude_interface.py."
            )

    elif detected_type == "llama":
        try:
            from .llama_interface import LlamaInterface
            return LlamaInterface(model_name=model_name)
        except ImportError:
            raise ImportError(
                f"Llama interface not available. Please create models/llama_interface.py."
            )

    elif detected_type == "deepseek":
        try:
            from .deepseek_interface import DeepSeekInterface
            return DeepSeekInterface(model_name=model_name)
        except ImportError:
            raise ImportError(
                f"DeepSeek interface not available. Please ensure models/deepseek_interface.py exists."
            )

    else:
        raise ValueError(
            f"Unsupported interface type: {detected_type}. "
            f"Please implement the corresponding interface class."
        )


def shutdown_backend_cache():
    """
    Shutdown and cleanup the global backend cache.

    Call this at the end of your program to properly release resources.
    """
    _global_backend_cache.shutdown()
