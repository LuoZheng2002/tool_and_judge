"""
vLLM backend with built-in automatic batching for concurrent inference.

This backend leverages vLLM's native asynchronous and batching capabilities,
which automatically handles request batching and efficient GPU utilization.
"""

import asyncio
from typing import Any, List, Optional
from .base import ModelBackend, ForwardResult, GenerationResult


class VLLMBackend(ModelBackend):
    """
    vLLM backend with automatic batching.

    vLLM provides native async inference with automatic continuous batching,
    so this backend is a thin wrapper that directly forwards requests to vLLM.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer: Any,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None
    ):
        """
        Initialize vLLM backend.

        Args:
            model_name: HuggingFace model name or path
            tokenizer: Tokenizer instance (for compatibility with interface)
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum model context length
        """
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.sampling_params import SamplingParams

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.SamplingParams = SamplingParams
        print(f"Initializing vLLM backend with tensor parallel size {tensor_parallel_size}...")
        # Create engine args
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True
        )

        # Initialize async engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Request counter for unique IDs
        self.request_counter = 0
        self.request_lock = asyncio.Lock()

    async def _get_next_request_id(self) -> str:
        """Generate a unique request ID."""
        async with self.request_lock:
            request_id = f"request_{self.request_counter}"
            self.request_counter += 1
            return request_id

    async def forward_async(
        self,
        prompt: str,
        max_length: int = 2048,
        **kwargs
    ) -> ForwardResult:
        """
        Forward pass is not supported in vLLM backend.

        vLLM is optimized for text generation and does not provide full logits
        tensors like HuggingFace. For tasks requiring forward passes (perplexity
        calculation, log probability comparison), use the HuggingFace backend.

        Raises:
            NotImplementedError: Always raised, as vLLM doesn't support forward pass
        """
        raise NotImplementedError(
            "Forward pass is not supported by vLLM backend. "
            "vLLM does not provide full logits tensors required for forward pass operations. "
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
        Asynchronously generate text from a formatted prompt.

        vLLM handles batching automatically, so we just submit the request.
        """
        request_id = await self._get_next_request_id()

        # Create sampling params with logprobs
        sampling_params = self.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=1.0 if not do_sample else 0.95,
            logprobs=1  # Request logprobs for generated tokens
        )

        # Submit request to vLLM engine
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id
        )

        # Wait for completion
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output is None:
            raise RuntimeError("vLLM generation returned no output")

        # Extract generated text and metadata
        output = final_output.outputs[0]
        generated_text = output.text.strip()
        generated_ids = output.token_ids

        # Get prompt token IDs
        prompt_ids = final_output.prompt_token_ids

        # Construct full sequence
        full_ids = prompt_ids + generated_ids
        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True).strip()

        # Extract logprobs (vLLM format: list of dicts)
        # output.logprobs is a list where each element corresponds to a generated token
        # Each element is a dict mapping token_id -> Logprob object
        logprobs = output.logprobs if hasattr(output, 'logprobs') else None

        result = GenerationResult(
            generated_text=generated_text,
            generated_ids=generated_ids,
            full_text=full_text,
            full_ids=full_ids,
            logits=logprobs  # vLLM provides logprobs, not full logits
        )

        return result

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
        """Get the tokenizer associated with this backend."""
        return self.tokenizer

    async def shutdown(self):
        """Cleanup resources and shutdown the backend."""
        # Properly shutdown the vLLM engine
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                # Shutdown the engine and wait for cleanup
                await self.engine.shutdown()
                print("vLLM engine shutdown completed")
            except Exception as e:
                print(f"Warning: Error during vLLM engine shutdown: {e}")
            finally:
                self.engine = None
