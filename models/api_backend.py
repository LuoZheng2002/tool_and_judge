"""
API-based backend for inference using OpenAI-compatible APIs.

This backend supports OpenAI's API and any other OpenAI-compatible API endpoints
(e.g., vLLM OpenAI server, local API servers).
"""

import asyncio
import os
from typing import Any, Optional
from .base import ModelBackend, ForwardResult, GenerationResult


class APIBackend(ModelBackend):
    """
    API backend for inference using OpenAI-compatible APIs.

    This backend makes API calls to OpenAI or OpenAI-compatible endpoints.
    It does not support forward pass operations (logits/perplexity calculation).
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize API backend.

        Args:
            model_name: Model identifier for the API (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: API key for authentication. If None, will try to get from OPENAI_API_KEY env var
            base_url: Custom API endpoint URL. If None, uses OpenAI's default endpoint
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "API backend requires the openai library. "
                "Install with: pip install openai"
            )

        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "API key is required for API backend. "
                "Provide it via api_key parameter or OPENAI_API_KEY environment variable."
            )

        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = AsyncOpenAI(**client_kwargs)

    async def forward_async(
        self,
        prompt: str,
        max_length: int = 2048,
        **kwargs
    ) -> ForwardResult:
        """
        Forward pass is not supported for API backends.

        API backends do not provide access to model logits, which are required
        for forward pass operations like perplexity calculation.

        Raises:
            NotImplementedError: Always raised, as API backends don't support forward pass
        """
        raise NotImplementedError(
            "Forward pass is not supported by API backends. "
            "API endpoints do not provide access to model logits required for forward pass operations. "
            "Please use 'huggingface' backend for tasks that require forward passes "
            "(e.g., perplexity calculation, direct preference comparison)."
        )

    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        Asynchronously generate text using the API.

        Args:
            prompt: The input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy decoding)
            do_sample: Whether to use sampling (affects temperature usage)
            **kwargs: Additional API-specific parameters

        Returns:
            GenerationResult containing generated text
        """
        # Prepare API parameters
        api_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
        }

        # Add any additional kwargs
        api_params.update(kwargs)

        try:
            # Make API call
            response = await self.client.chat.completions.create(**api_params)

            # Extract generated text
            generated_text = response.choices[0].message.content.strip()

            # For API backends, we don't have token IDs or full sequence info
            # Return placeholder values
            result = GenerationResult(
                generated_text=generated_text,
                generated_ids=[],  # Not available from API
                full_text=prompt + "\n" + generated_text,  # Approximate
                full_ids=[],  # Not available from API
                logits=None  # Not available from API
            )

            return result

        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}") from e

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs
    ) -> GenerationResult:
        """Synchronous version of generate_async."""
        return asyncio.run(
            self.generate_async(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs
            )
        )

    def get_tokenizer(self) -> Any:
        """
        Get tokenizer is not supported for API backends.

        API backends do not provide direct access to tokenizers.

        Raises:
            NotImplementedError: Always raised, as API backends don't expose tokenizers
        """
        raise NotImplementedError(
            f"APIBackend does not provide direct tokenizer access. "
            f"Tokenization is handled internally by the API service."
        )

    async def shutdown(self):
        """Cleanup resources and shutdown the backend."""
        # Close the HTTP client
        await self.client.close()
