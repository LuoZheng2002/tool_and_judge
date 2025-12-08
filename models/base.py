"""
Unified base classes for model interfaces and backends.

This module defines the abstract interfaces that all model-specific implementations
must follow. It separates the model interface (handling task-specific logic) from
the backend (handling actual inference).

Key Design Principles:
- ModelInterface exposes only HIGH-LEVEL TASK methods (tool calling, comparison, perplexity)
- NO intermediate methods like build_prompt (API vs local models differ too much)
- Each model implements tasks however they want (messages API vs chat templates, etc.)
- ModelBackend handles inference execution (API calls, local GPU inference)
- Backends are cached and reused across different configurations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import asyncio


from models.name_mapping import FunctionNameMapper
from config import PostprocessError


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class ForwardResult:
    """Result from a forward pass inference (logits computation)."""

    logits: Any  # Tensor of shape [seq_len, vocab_size]
    input_ids: List[int]  # List of token IDs that were input to the model

    def __init__(self, logits: Any, input_ids: List[int]):
        """
        Args:
            logits: Tensor of shape [seq_len, vocab_size] containing logits for each position
            input_ids: List of token IDs that were input to the model
        """
        self.logits = logits
        self.input_ids = input_ids


@dataclass
class GenerationResult:
    """Result from a text generation inference."""

    generated_text: str  # The decoded generated text (without input prompt)
    generated_ids: List[int]  # List of token IDs for the generated text only
    full_text: str  # The complete text (prompt + generated)
    full_ids: List[int]  # List of token IDs for the complete sequence
    logits: Optional[List[Dict[int, float]]] = None  # Unified logprobs format

    def __init__(
        self,
        generated_text: str,
        generated_ids: List[int],
        full_text: str,
        full_ids: List[int],
        logits: Optional[List[Dict[int, float]]] = None
    ):
        """
        Args:
            generated_text: The decoded generated text (without input prompt)
            generated_ids: List of token IDs for the generated text only
            full_text: The complete text (prompt + generated)
            full_ids: List of token IDs for the complete sequence (prompt + generated)
            logits: Unified log probability format for generated tokens:
                    - List length matches number of generated tokens
                    - Each element is a Dict[int, float] mapping token_id -> log_probability
                    - None if logprobs were not requested or backend doesn't support them

                    Example for 2 generated tokens:
                    [
                        {101: -0.5, 102: -1.2, 103: -2.0, ...},  # Token 1 logprobs
                        {201: -0.3, 202: -1.5, 203: -1.8, ...}   # Token 2 logprobs
                    ]
        """
        self.generated_text = generated_text
        self.generated_ids = generated_ids
        self.full_text = full_text
        self.full_ids = full_ids
        self.logits = logits


@dataclass
class ComparisonResult:
    """Result from a preference comparison task (deprecated - use DirectComparisonResult or CoTComparisonResult)."""

    preference: int  # Which answer is preferred (1 or 2)
    reasoning: Optional[str] = None  # Optional reasoning text (for CoT)
    raw_output: Optional[str] = None  # Full model output

    def __init__(
        self,
        preference: int,
        reasoning: Optional[str] = None,
        raw_output: Optional[str] = None
    ):
        """
        Args:
            preference: Which answer is preferred (1 or 2)
            reasoning: Optional reasoning text for chain-of-thought comparisons
            raw_output: Full raw output from the model
        """
        self.preference = preference
        self.reasoning = reasoning
        self.raw_output = raw_output


@dataclass
class DirectComparisonResult:
    """Result from a direct preference comparison (without reasoning)."""

    preference: Optional[int]  # Which answer is preferred (1 or 2), None if error
    raw_output: str  # Full model output (reasoning merged if any)
    logit_1: float  # Probability/logit for choice "1" (mandatory)
    logit_2: float  # Probability/logit for choice "2" (mandatory)
    error: Optional[str] = None  # Error message if model fails to choose

    def __init__(
        self,
        preference: Optional[int],
        raw_output: str,
        logit_1: float,
        logit_2: float,
        error: Optional[str] = None
    ):
        """
        Args:
            preference: Which answer is preferred (1 or 2), None if error occurred
            raw_output: Full raw output from the model (reasoning merged if any)
            logit_1: Probability/logit for choice "1" (mandatory)
            logit_2: Probability/logit for choice "2" (mandatory)
            error: Error message if the model failed to choose between 1 and 2
        """
        self.preference = preference
        self.raw_output = raw_output
        self.logit_1 = logit_1
        self.logit_2 = logit_2
        self.error = error


@dataclass
class CoTComparisonResult:
    """Result from a chain-of-thought preference comparison (with reasoning)."""

    preference: Optional[int] = None  # Which answer is preferred (1 or 2), None if error
    raw_output: str = ""  # Full model output (reasoning merged with final answer)
    error: Optional[str] = None  # Error message if model fails to choose

    def __init__(
        self,
        preference: Optional[int] = None,
        raw_output: str = "",
        error: Optional[str] = None
    ):
        """
        Args:
            preference: Which answer is preferred (1 or 2), None if error occurred
            raw_output: Full raw output from the model (reasoning merged with final answer)
            error: Error message if the model failed to choose between 1 and 2
        """
        self.preference = preference
        self.raw_output = raw_output
        self.error = error


# =============================================================================
# Model Backend Abstract Base Class
# =============================================================================

class ModelBackend(ABC):
    """
    Abstract base class for model backends that handle inference.

    Different backends (API, HuggingFace, vLLM) implement this interface to provide
    low-level inference primitives. The backend's job is ONLY to execute inference,
    not to format prompts or parse outputs (that's the interface's job).

    Backends should automatically batch concurrent async requests for efficiency.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        return_logprobs: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        Synchronously generate text from a prompt.

        Args:
            prompt: The input prompt text (already formatted by interface)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 for greedy decoding, >0 for sampling)
            return_logprobs: If True, return log probabilities for generated tokens.
                           Backend MUST provide logprobs when this is True, or raise error.
            **kwargs: Additional backend-specific parameters

        Returns:
            GenerationResult containing generated text and token IDs.
            If return_logprobs=True, GenerationResult.logits will contain
            List[Dict[int, float]] with log probabilities.

        Raises:
            RuntimeError: If return_logprobs=True but backend fails to provide logprobs
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        return_logprobs: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        Asynchronously generate text from a prompt.

        The backend may batch multiple concurrent requests internally for efficiency.

        Args:
            prompt: The input prompt text (already formatted by interface)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 for greedy decoding, >0 for sampling)
            return_logprobs: If True, return log probabilities for generated tokens.
                           Backend MUST provide logprobs when this is True, or raise error.
            **kwargs: Additional backend-specific parameters

        Returns:
            GenerationResult containing generated text and token IDs.
            If return_logprobs=True, GenerationResult.logits will contain
            List[Dict[int, float]] with log probabilities.

        Raises:
            RuntimeError: If return_logprobs=True but backend fails to provide logprobs
        """
        pass

    @abstractmethod
    async def forward_async(
        self,
        prompt: str,
        max_length: int = 2048,
        **kwargs
    ) -> ForwardResult:
        """
        Asynchronously run forward pass on a prompt to get logits.

        This method is used for perplexity calculation and other tasks that
        require access to the model's output logits.

        Args:
            prompt: The input prompt text (already formatted by interface)
            max_length: Maximum sequence length for tokenization
            **kwargs: Additional backend-specific parameters

        Returns:
            ForwardResult containing logits and input_ids

        Raises:
            NotImplementedError: If backend doesn't support forward pass (e.g., API backends)
        """
        pass

    @abstractmethod
    async def shutdown(self):
        """
        Cleanup resources and shutdown the backend.

        Should be called when inference is complete to properly release resources.
        """
        pass

    def get_tokenizer(self) -> Any:
        """
        Get the tokenizer associated with this backend.

        Returns:
            Tokenizer object (e.g., HuggingFace tokenizer)

        Raises:
            NotImplementedError: If backend doesn't support direct tokenizer access
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide direct tokenizer access"
        )


# =============================================================================
# Model Interface Abstract Base Class
# =============================================================================

class ModelInterface(ABC):
    """
    Abstract base class for model-specific behavior.

    This interface exposes ONLY high-level task methods. Each model implements
    these tasks however they want (API calls, chat templates, etc.).

    Key principle: NO intermediate methods like build_prompt or parse_output.
    Each task method handles its own prompt formatting and output parsing internally.
    """

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the model name/identifier.

        Returns:
            Model name string (e.g., "gpt-5", "Qwen/Qwen3-8B")
        """
        pass


# =============================================================================
# Tool Model Interface (for Tool Project)
# =============================================================================

class ToolModelInterface(ModelInterface):
    """
    Interface for tool project with function calling capabilities.

    Exposes high-level methods for:
    - Generating function calls from user queries
    - Preprocessing function definitions
    - Postprocessing model outputs

    For batch processing, use asyncio.gather() at the call site:
        tasks = [interface.generate_tool_call_async(backend, funcs, query)
                 for funcs, query in zip(functions_list, queries)]
        results = await asyncio.gather(*tasks)
    """

    @abstractmethod
    async def generate_tool_call_async(
        self,
        backend: ModelBackend,
        raw_functions: List[Dict[str, Any]],
        user_query: str,
        name_mapper: FunctionNameMapper,
        prompt_passing_in_english: bool,
        max_new_tokens: int = 512,
        temperature: float = 0.0,        
    ) -> str:
        """
        Generate tool/function calls from a user query.

        This is the main entry point for tool calling. The method should:
        1. Preprocess functions (sanitization, schema fixing, etc.)
        2. Format the prompt (API messages, chat template, etc.)
        3. Call the backend to generate
        4. Return raw output (postprocessing done separately)

        Args:
            backend: The backend to use for inference
            functions: List of available function definitions
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters

        Returns:
            Raw model output as a string
        """
        pass

    def generate_tool_call(
        self,
        backend: ModelBackend,
        raw_functions: List[Dict[str, Any]],
        user_query: str,
        prompt_passing_in_english: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Synchronous version of generate_tool_call_async.

        Default implementation uses asyncio.run to call async version.
        """
        return asyncio.run(
            self.generate_tool_call_async(
                backend=backend,
                raw_functions=raw_functions,
                user_query=user_query,
                prompt_passing_in_english=prompt_passing_in_english,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
        )

    # def preprocess_functions(
    #     self,
    #     functions: List[Dict[str, Any]],
    #     name_mapper: Optional['FunctionNameMapper']
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Preprocess function definitions before passing to model.

    #     This is where you can:
    #     - Sanitize function names (e.g., GPT-5 requires alphanumeric)
    #     - Fix JSON schemas (e.g., convert "dict" to "object")
    #     - Add prompt instructions
    #     - Use name_mapper.get_sanitized_name() to sanitize and cache mappings

    #     Default implementation returns functions unchanged.

    #     Args:
    #         functions: List of function definitions
    #         name_mapper: FunctionNameMapper instance for sanitizing names
    #                     (automatically caches mappings on get_sanitized_name() calls)

    #     Returns:
    #         Preprocessed function definitions
    #     """
    #     return functions

    @abstractmethod
    def postprocess_tool_calls(
        self,
        raw_output: str,
        name_mapper: Optional['FunctionNameMapper'] = None
    ) -> Union[List[Dict[str, Dict[str, Any]]], Tuple[PostprocessError, Dict[str, Any]]]:
        """
        Postprocess raw model output to extract function calls.

        This should parse the model's output and extract function calls
        in a standardized format.

        Args:
            raw_output: Raw string output from the model
            name_mapper: FunctionNameMapper instance to convert sanitized names back
                        to original using get_original_name()

        Returns:
            On success: List of function call dictionaries in format:
                [
                    {
                        "function_name": {
                            "param1": value1,
                            "param2": value2,
                            ...
                        }
                    },
                    ...
                ]
            On error: Tuple of (PostprocessError, metadata_dict) where metadata_dict
                contains error details (e.g., raw_output, error_message, exception
                details, etc.)

        Examples:
            >>> from models.name_mapping import FunctionNameMapper
            >>> name_mapper = FunctionNameMapper()
            >>> raw_output = '<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>'
            >>> result = interface.postprocess_tool_calls(raw_output, name_mapper)
            >>> isinstance(result, list)
            True
            >>> result
            [{"get_weather": {"location": "Paris"}}]

            >>> # Error case
            >>> raw_output = 'invalid json'
            >>> result = interface.postprocess_tool_calls(raw_output, name_mapper)
            >>> isinstance(result, tuple)
            True
            >>> error, metadata = result
            >>> error == PostprocessError.JSON_DECODE_ERROR
            True
            >>> 'raw_output' in metadata
            True
        """
        pass

    @abstractmethod
    async def translate_tool_question_async(
        self,
        backend: ModelBackend,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Translate a user question to English.

        This method takes only the question text (without function definitions or JSON)
        and translates it to English.

        Args:
            backend: The backend to use for inference
            question: The question text to translate
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Translated question as a string
        """
        pass

    @abstractmethod
    async def translate_tool_answer_async(
        self,
        backend: ModelBackend,
        parameter_value: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """
        Translate a single function parameter value to English.

        This method translates individual parameter values one at a time.

        Args:
            backend: The backend to use for inference
            parameter_value: The parameter value to translate
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Translated parameter value as a string
        """
        pass


# =============================================================================
# Judge Model Interface (for Judge Project)
# =============================================================================

class JudgeModelInterface(ModelInterface):
    """
    Interface for judge project with perplexity and preference comparison.

    Exposes high-level methods for:
    - Preference comparison (direct and with reasoning)
    - Perplexity calculation (forward pass with logits)
    """

    @abstractmethod
    async def compare_directly_async(
        self,
        backend: ModelBackend,
        question: str,
        answer1: str,
        answer2: str,
        **kwargs
    ) -> 'DirectComparisonResult':
        """
        Compare two answers directly without reasoning.

        This method should:
        1. Format the comparison prompt (with question and both answers)
        2. Call backend to generate with logprobs enabled
        3. Parse the output to extract preference (1 or 2)
        4. Extract probability logits for choices "1" and "2" (MANDATORY - must raise error if unavailable)
        5. Return DirectComparisonResult with preference, logit_1, logit_2, and optional error

        IMPORTANT: logit_1 and logit_2 are mandatory fields. If the backend fails to provide
        logits for both choices, the implementation MUST raise an error and exit immediately.

        Args:
            backend: The backend to use for inference
            question: The question being answered
            answer1: First answer to compare
            answer2: Second answer to compare
            **kwargs: Additional model-specific parameters

        Returns:
            DirectComparisonResult with preference (1 or 2), logit_1, logit_2, and optional error

        Raises:
            RuntimeError: If logits cannot be extracted from backend response
        """
        pass

    def compare_directly(
        self,
        backend: ModelBackend,
        question: str,
        answer1: str,
        answer2: str,
        **kwargs
    ) -> 'DirectComparisonResult':
        """Synchronous version of compare_directly_async."""
        return asyncio.run(
            self.compare_directly_async(
                backend=backend,
                question=question,
                answer1=answer1,
                answer2=answer2,
                **kwargs
            )
        )

    @abstractmethod
    async def compare_thinking_async(
        self,
        backend: ModelBackend,
        question: str,
        answer1: str,
        answer2: str,
        **kwargs
    ) -> 'CoTComparisonResult':
        """
        Compare two answers with chain-of-thought reasoning.

        This method should:
        1. Format the comparison prompt (encouraging reasoning)
        2. Call backend to generate (with reasoning enabled if applicable)
        3. Parse the output to extract both reasoning and preference
        4. Merge reasoning with raw output
        5. Return CoTComparisonResult with merged output and optional error

        Args:
            backend: The backend to use for inference
            question: The question being answered
            answer1: First answer to compare
            answer2: Second answer to compare
            **kwargs: Additional model-specific parameters

        Returns:
            CoTComparisonResult with preference (1 or 2), merged raw output, and optional error
        """
        pass

    def compare_thinking(
        self,
        backend: ModelBackend,
        question: str,
        answer1: str,
        answer2: str,
        **kwargs
    ) -> 'CoTComparisonResult':
        """Synchronous version of compare_thinking_async."""
        return asyncio.run(
            self.compare_thinking_async(
                backend=backend,
                question=question,
                answer1=answer1,
                answer2=answer2,
                **kwargs
            )
        )

    @abstractmethod
    async def forward_for_logits_async(
        self,
        backend: ModelBackend,
        question: str,
        answer: str,
        language: str = "English",
        **kwargs
    ) -> ForwardResult:
        """
        Run forward pass to get logits for perplexity calculation.

        This method should:
        1. Format the prompt with question and answer
        2. Call backend.forward_async to get logits
        3. Return ForwardResult with logits and input_ids

        Note: This is only supported for local model backends.
        API backends should raise NotImplementedError.

        Args:
            backend: The backend to use for inference
            question: The question
            answer: The answer to calculate perplexity for
            language: Language name (e.g., "English", "Chinese")
            **kwargs: Additional model-specific parameters

        Returns:
            ForwardResult containing logits and input_ids

        Raises:
            NotImplementedError: If model doesn't support forward pass (API models)
        """
        pass

    def forward_for_logits(
        self,
        backend: ModelBackend,
        question: str,
        answer: str,
        language: str = "English",
        **kwargs
    ) -> ForwardResult:
        """Synchronous version of forward_for_logits_async."""
        return asyncio.run(
            self.forward_for_logits_async(
                backend=backend,
                question=question,
                answer=answer,
                language=language,
                **kwargs
            )
        )

    def get_assistant_prefix(self) -> str:
        """
        Get the assistant prefix token/string for finding answer positions.

        This is used to locate where the answer starts in the tokenized sequence
        for perplexity calculation. Only needed for local models.

        Returns:
            Assistant prefix string

        Raises:
            NotImplementedError: If not applicable for this model (e.g., API models)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not use assistant prefix tokens"
        )

    def find_answer_start(
        self,
        tokenizer: Any,
        full_ids: List[int],
        answer_tokens: List[int]
    ) -> int:
        """
        Find the starting position of the answer tokens in the full sequence.

        Default implementation uses the assistant prefix to locate where
        the answer begins. Override if your model needs custom logic.

        Args:
            tokenizer: The model's tokenizer
            full_ids: List of all token IDs in the full conversation
            answer_tokens: List of token IDs for just the answer

        Returns:
            Index of the first answer token in full_ids

        Raises:
            ValueError: If the answer start position cannot be found
        """
        assistant_prefix = self.get_assistant_prefix()
        prefix_ids = tokenizer(assistant_prefix, add_special_tokens=False).input_ids

        # Find the prefix in the full sequence
        L = len(prefix_ids)
        for i in range(len(full_ids) - L + 1):
            if full_ids[i:i+L] == prefix_ids:
                return i + L  # Return position after the prefix

        raise ValueError(
            f"Assistant prefix '{assistant_prefix}' not found in tokenized conversation. "
            f"This might indicate a mismatch between the model interface and the actual model."
        )
