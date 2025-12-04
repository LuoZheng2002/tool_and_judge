"""
HuggingFace Transformers backend with manual batching for concurrent inference.

This backend accepts single async requests and automatically batches them
for efficient processing on GPU.
"""

import asyncio
from typing import Any, List
from .base import ModelBackend, ForwardResult, GenerationResult


class HuggingFaceBackend(ModelBackend):
    """
    HuggingFace Transformers backend with manual dynamic batching.

    This backend collects concurrent requests and batches them together
    for efficient GPU processing. Requests are accumulated for a short
    time window (max_batch_wait) or until the batch size reaches max_batch_size.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        max_batch_size: int = 8,
        max_batch_wait: float = 0.05  # 50ms wait time
    ):
        """
        Initialize HuggingFace backend.

        Args:
            model: HuggingFace model instance
            tokenizer: HuggingFace tokenizer instance
            device: Device to run inference on
            max_batch_size: Maximum batch size for inference
            max_batch_wait: Maximum time (seconds) to wait for batch accumulation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_batch_wait = max_batch_wait

        # Queues for batching requests
        self.forward_queue: List[tuple] = []
        self.generation_queue: List[tuple] = []

        # Locks for thread-safe queue operations
        self.forward_lock = asyncio.Lock()
        self.generation_lock = asyncio.Lock()

        # Background tasks for processing batches
        self.forward_task = None
        self.generation_task = None
        self.running = True

        # Save original padding side
        self.original_padding_side = tokenizer.padding_side

    async def _start_batch_processors(self):
        """Start background tasks for processing batches."""
        if self.forward_task is None:
            self.forward_task = asyncio.create_task(self._process_forward_batches())
        if self.generation_task is None:
            self.generation_task = asyncio.create_task(self._process_generation_batches())

    async def _process_forward_batches(self):
        """Background task that processes forward pass batches."""
        import torch

        while self.running:
            await asyncio.sleep(self.max_batch_wait)

            async with self.forward_lock:
                if not self.forward_queue:
                    continue

                # Get batch (up to max_batch_size)
                batch = self.forward_queue[:self.max_batch_size]
                self.forward_queue = self.forward_queue[self.max_batch_size:]

            if not batch:
                continue

            # Extract requests from batch
            prompts = [item[0] for item in batch]
            max_lengths = [item[1] for item in batch]
            futures = [item[2] for item in batch]

            try:
                # Use the maximum max_length from the batch
                max_length = max(max_lengths)

                # Set padding side to right for forward pass
                self.tokenizer.padding_side = 'right'

                # Tokenize batch
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Run forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                # Set results for each request
                for i, future in enumerate(futures):
                    result = ForwardResult(
                        logits=logits[i].cpu(),  # Move to CPU to free GPU memory
                        input_ids=inputs['input_ids'][i].cpu().tolist()
                    )
                    future.set_result(result)

            except Exception as e:
                # Set exception for all requests in batch
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

            finally:
                # Restore original padding side
                self.tokenizer.padding_side = self.original_padding_side

    async def _process_generation_batches(self):
        """Background task that processes generation batches."""
        import torch

        while self.running:
            await asyncio.sleep(self.max_batch_wait)

            async with self.generation_lock:
                if not self.generation_queue:
                    continue

                # Get batch (up to max_batch_size)
                batch = self.generation_queue[:self.max_batch_size]
                self.generation_queue = self.generation_queue[self.max_batch_size:]

            if not batch:
                continue

            # Extract requests from batch
            prompts = [item[0] for item in batch]
            max_new_tokens_list = [item[1] for item in batch]
            temperatures = [item[2] for item in batch]
            do_samples = [item[3] for item in batch]
            return_logprobs_list = [item[4] for item in batch]
            futures = [item[5] for item in batch]

            try:
                # Use max values from batch
                max_new_tokens = max(max_new_tokens_list)
                # For temperature and do_sample, use the most common value or first value
                temperature = temperatures[0]
                do_sample = do_samples[0]
                # Check if any request needs logprobs
                any_return_logprobs = any(return_logprobs_list)

                # Set padding side to left for generation
                self.tokenizer.padding_side = 'left'

                # Tokenize batch
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate with scores to get logits (only if requested)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if do_sample else 1.0,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=any_return_logprobs  # Only compute scores if needed
                    )

                output_ids = outputs.sequences
                scores = outputs.scores if any_return_logprobs else None  # Tuple of tensors, one per generated token

                # Decode results for each request
                for i, future in enumerate(futures):
                    input_length = inputs['input_ids'][i].shape[0]
                    return_logprobs = return_logprobs_list[i]

                    # Extract generated tokens
                    generated_ids = output_ids[i][input_length:].cpu().tolist()
                    generated_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True
                    ).strip()

                    # Extract full sequence
                    full_ids = output_ids[i].cpu().tolist()
                    full_text = self.tokenizer.decode(
                        full_ids,
                        skip_special_tokens=True
                    ).strip()

                    # Convert logits to unified format if requested
                    unified_logits = None
                    if return_logprobs:
                        if scores is None or len(scores) == 0:
                            # Logprobs were requested but not available
                            future.set_exception(RuntimeError(
                                "HuggingFace backend failed to provide logprobs. "
                                "return_logprobs=True was specified but scores are not available. "
                                "This may indicate a backend configuration issue."
                            ))
                            continue

                        # Convert HuggingFace scores to unified format
                        # scores is a tuple of [batch_size, vocab_size] tensors
                        unified_logits = []
                        for score_tensor in scores:
                            # Get logits for this sample (i-th element in batch)
                            token_logits = score_tensor[i].cpu()

                            # Apply log_softmax to convert logits to log probabilities
                            log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)

                            # Convert to dict format: {token_id: log_prob}
                            token_dict = {
                                token_id: float(log_probs[token_id])
                                for token_id in range(len(log_probs))
                            }
                            unified_logits.append(token_dict)

                        # Verify length matches generated tokens
                        if len(unified_logits) != len(generated_ids):
                            future.set_exception(RuntimeError(
                                f"Logprobs length mismatch: expected {len(generated_ids)} tokens, "
                                f"but got {len(unified_logits)} logprob entries."
                            ))
                            continue

                    result = GenerationResult(
                        generated_text=generated_text,
                        generated_ids=generated_ids,
                        full_text=full_text,
                        full_ids=full_ids,
                        logits=unified_logits
                    )
                    future.set_result(result)

            except Exception as e:
                # Set exception for all requests in batch
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

            finally:
                # Restore original padding side
                self.tokenizer.padding_side = self.original_padding_side

    async def forward_async(
        self,
        prompt: str,
        max_length: int = 2048,
        **kwargs
    ) -> ForwardResult:
        """
        Asynchronously run forward pass on a formatted prompt.

        Adds the request to a queue that will be batched with other concurrent requests.
        """
        # Start batch processors if not already running
        await self._start_batch_processors()

        # Create a future for this request
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        # Add request to queue
        async with self.forward_lock:
            self.forward_queue.append((prompt, max_length, future))

        # Wait for result
        return await future

    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        do_sample: bool = False,
        return_logprobs: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        Asynchronously generate text from a formatted prompt.

        Adds the request to a queue that will be batched with other concurrent requests.

        Args:
            prompt: The input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            return_logprobs: If True, return log probabilities in unified format

        Returns:
            GenerationResult with unified logits format if return_logprobs=True

        Raises:
            RuntimeError: If return_logprobs=True but backend fails to provide logprobs
        """
        # Start batch processors if not already running
        await self._start_batch_processors()

        # Create a future for this request
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        # Add request to queue
        async with self.generation_lock:
            self.generation_queue.append((
                prompt,
                max_new_tokens,
                temperature,
                do_sample,
                return_logprobs,
                future
            ))

        # Wait for result
        return await future

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        do_sample: bool = False,
        return_logprobs: bool = False,
        **kwargs
    ) -> GenerationResult:
        """Synchronous version of generate_async."""
        return asyncio.run(
            self.generate_async(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                return_logprobs=return_logprobs,
                **kwargs
            )
        )

    def get_tokenizer(self) -> Any:
        """Get the tokenizer associated with this backend."""
        return self.tokenizer

    async def shutdown(self):
        """Cleanup resources and shutdown the backend."""
        self.running = False

        # Cancel background tasks
        if self.forward_task:
            self.forward_task.cancel()
            try:
                await self.forward_task
            except asyncio.CancelledError:
                pass

        if self.generation_task:
            self.generation_task.cancel()
            try:
                await self.generation_task
            except asyncio.CancelledError:
                pass

        # Restore original padding side
        self.tokenizer.padding_side = self.original_padding_side
