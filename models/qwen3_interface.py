"""
Unified Qwen3 interface for both tool and judge projects.

This interface supports:
- Tool project: Function calling with <tool_call> format
- Judge project: Perplexity calculation and preference comparison

Qwen3 models use ChatML format with:
- Special tokens: <|im_start|> and <|im_end|>
- Thinking mode: <think></think> tags for chain-of-thought reasoning
- Tool calling: <tool_call>{...}</tool_call> format
"""

import json
import re
from typing import List, Dict, Any, Union, Optional, TYPE_CHECKING
from .base import (
    JudgeModelInterface,
    ToolModelInterface,
    ModelBackend,
    ComparisonResult,
    ForwardResult,
)

if TYPE_CHECKING:
    from .name_mapping import FunctionNameMapper


class Qwen3Interface(JudgeModelInterface, ToolModelInterface):
    """
    Unified interface for Qwen3 models supporting both tool and judge use cases.

    This interface inherits from both JudgeModelInterface and ToolModelInterface,
    providing functionality for:
    - Function calling (tool project)
    - Perplexity calculation (judge project)
    - Preference comparison (judge project)
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-8B", enable_thinking: bool = False):
        """
        Initialize the Qwen3 interface.

        Args:
            model_name: Model identifier (e.g., "Qwen/Qwen3-8B", "Qwen/Qwen3-14B")
            enable_thinking: Whether to enable chain-of-thought reasoning mode
        """
        self.model_name = model_name
        self.enable_thinking = enable_thinking

    # =========================================================================
    # ModelInterface Methods
    # =========================================================================

    def get_model_name(self) -> str:
        """Get the model name/identifier."""
        return self.model_name

    # =========================================================================
    # ToolModelInterface Methods
    # =========================================================================

    async def generate_tool_call_async(
        self,
        backend: ModelBackend,
        functions: List[Dict[str, Any]],
        user_query: str,
        prompt_passing_in_english: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate tool/function calls from a user query.

        This method:
        1. Builds system prompt with function definitions
        2. Formats the full prompt with ChatML + <tools> section
        3. Calls backend to generate
        4. Returns raw output (with <tool_call> tags)

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
        # Build system prompt
        passing_in_english_prompt = (
            " IMPORTANT: Pass in all parameters in function calls in English."
            if prompt_passing_in_english
            else ""
        )

        system_prompt = f'''You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should ONLY return function calls in your response. You MUST NOT include any other text, explanations, or direct answers. If you decide to invoke any function(s), you MUST use the provided tools. Do NOT attempt to answer the question directly without using the available functions.{passing_in_english_prompt}

You should only return the function calls in your response, in JSON format as a list where each element has the format {{"name": "function_name", "arguments": {{param1: value1, param2: value2, ...}}}}.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.'''

        # Build full prompt with tools section
        formatted_prompt = ""

        # Add system message with tools
        formatted_prompt += "<|im_start|>system\n"
        formatted_prompt += system_prompt + "\n\n"
        formatted_prompt += (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>"
        )
        for func in functions:
            formatted_prompt += "\n" + json.dumps(func, ensure_ascii=False)
        formatted_prompt += (
            "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call><|im_end|>\n"
        )

        # Add user message
        formatted_prompt += f"<|im_start|>user\n{user_query}<|im_end|>\n"

        # Add generation prompt
        formatted_prompt += "<|im_start|>assistant\n"

        # Disable thinking mode by default unless enabled
        if not self.enable_thinking:
            formatted_prompt += "<think>\n\n</think>\n\n"

        # Call backend
        result = await backend.generate_async(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            **kwargs
        )

        return result.generated_text

    def postprocess_tool_calls(
        self,
        raw_output: str,
        name_mapper: Optional['FunctionNameMapper'] = None
    ) -> Union[List[Dict[str, Dict[str, Any]]], str]:
        """
        Postprocess raw output from Qwen3 model to extract function calls.

        Qwen3 outputs in JSON format within <tool_call> tags:
        <tool_call>
        {"name": "function_name", "arguments": {"param1": value1, ...}}
        </tool_call>

        Note: Qwen3 doesn't require name sanitization, so name_mapper is unused.

        Args:
            raw_output: Raw string output from the model
            name_mapper: Unused for Qwen3 (no name sanitization needed)

        Returns:
            List of function call dictionaries in format: [{func_name: {arguments}}, ...]
            Returns error string if parsing fails
        """
        # Parse Qwen3 model's output format: <tool_call>{...}</tool_call>
        model_result_raw = raw_output.strip()

        # Remove reasoning content if present (content between <think></think> tags)
        if "<think>" in model_result_raw and "</think>" in model_result_raw:
            # Extract only the content after </think>
            think_end_idx = model_result_raw.find("</think>")
            if think_end_idx != -1:
                model_result_raw = model_result_raw[think_end_idx + len("</think>"):].strip()

        # Extract content from <tool_call> tags if present
        if "<tool_call>" in model_result_raw:
            start_idx = model_result_raw.find("<tool_call>")
            end_idx = model_result_raw.find("</tool_call>")
            if start_idx != -1 and end_idx != -1:
                model_result_raw = model_result_raw[start_idx + len("<tool_call>"):end_idx]
                model_result_raw = model_result_raw.strip()

        # Strip backticks and whitespace
        model_result_raw = model_result_raw.strip("`\n ")

        # Add brackets if missing (for single objects or arrays)
        if not model_result_raw.startswith("["):
            # Try to parse as single JSON object first
            if model_result_raw.startswith("{"):
                model_result_raw = "[" + model_result_raw + "]"
            else:
                model_result_raw = "[" + model_result_raw
        if not model_result_raw.endswith("]"):
            model_result_raw = model_result_raw + "]"

        try:
            # Parse the JSON array
            tool_calls = json.loads(model_result_raw)
        except json.JSONDecodeError:
            return f"Failed to decode JSON: Invalid JSON format. Raw string: {model_result_raw}"

        # Convert Qwen3 format to desired format
        extracted = []
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                    func_name = tool_call["name"]
                    func_args = tool_call["arguments"]
                    extracted.append({func_name: func_args})
                else:
                    return f"Failed to decode JSON: Invalid tool call structure. Raw string: {model_result_raw}"
        else:
            return f"Failed to decode JSON: Expected a list of tool calls. Raw string: {model_result_raw}"

        return extracted

    # =========================================================================
    # JudgeModelInterface Methods
    # =========================================================================

    async def compare_directly_async(
        self,
        backend: ModelBackend,
        question: str,
        answer1: str,
        answer2: str,
        **kwargs
    ) -> ComparisonResult:
        """
        Compare two answers directly without reasoning.

        This method:
        1. Formats the comparison prompt
        2. Calls backend to generate
        3. Parses output to extract preference (1 or 2)

        Args:
            backend: The backend to use for inference
            question: The question being answered
            answer1: First answer to compare
            answer2: Second answer to compare
            **kwargs: Additional model-specific parameters

        Returns:
            ComparisonResult with preference (1 or 2)
        """
        # Build comparison prompt
        prompt_text = f"""Given the following question and two answers, which answer is better?

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Provide your judgment IMMEDIATELY without reasoning or explanation. Provide your final decision in the following format:
\\boxed{{X}} where X is either 1 or 2."""

        # Format with ChatML
        formatted_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"

        # Disable thinking for direct comparison
        formatted_prompt += "<think>\n\n</think>\n\n"

        # Call backend
        result = await backend.generate_async(
            prompt=formatted_prompt,
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
            **kwargs
        )

        # Parse preference from output
        raw_output = result.generated_text
        preference = self._parse_preference(raw_output)

        return ComparisonResult(
            preference=preference,
            reasoning=None,
            raw_output=raw_output
        )

    async def compare_thinking_async(
        self,
        backend: ModelBackend,
        question: str,
        answer1: str,
        answer2: str,
        **kwargs
    ) -> ComparisonResult:
        """
        Compare two answers with chain-of-thought reasoning.

        This method:
        1. Formats the comparison prompt (encouraging reasoning)
        2. Calls backend with thinking enabled
        3. Parses output to extract reasoning and preference

        Args:
            backend: The backend to use for inference
            question: The question being answered
            answer1: First answer to compare
            answer2: Second answer to compare
            **kwargs: Additional model-specific parameters

        Returns:
            ComparisonResult with preference (1 or 2) and reasoning text
        """
        # Build comparison prompt with CoT instruction
        prompt_text = f"""Given the following question and two answers, which answer is better?

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Please briefly explain your reasoning, and then provide your final decision in the following format:
\\boxed{{X}} where X is either 1 or 2."""

        # Format with ChatML (thinking enabled)
        formatted_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"

        # Don't add empty <think> tags - let model use them if it wants
        # (or it will reason in regular text)

        # Call backend
        result = await backend.generate_async(
            prompt=formatted_prompt,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            **kwargs
        )

        # Parse preference and extract reasoning
        raw_output = result.generated_text
        preference = self._parse_preference(raw_output)

        # Extract reasoning (text before the final \\boxed{} decision)
        reasoning = self._extract_reasoning(raw_output)

        return ComparisonResult(
            preference=preference,
            reasoning=reasoning,
            raw_output=raw_output
        )

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

        This method:
        1. Gets tokenizer from backend
        2. Formats the prompt with question and answer
        3. Applies chat template using tokenizer
        4. Calls backend.forward_async to get logits

        Args:
            backend: The backend to use for inference
            question: The question
            answer: The answer to calculate perplexity for
            language: Language name (e.g., "English", "Chinese")
            **kwargs: Additional model-specific parameters

        Returns:
            ForwardResult containing logits and input_ids
        """
        # Get tokenizer from backend
        tokenizer = backend.get_tokenizer()

        # Build language-specific instructions
        if language.lower() == "english":
            instruction = "Please answer the question in English with a concise phrase instead of a complete sentence. Start with an uncapitalized first word."
        else:
            instruction = f"Please answer the question in {language} with a concise phrase instead of a complete sentence."

        # Combine question with instruction
        user_content = f"{question}\n\n{instruction}"

        # Build messages
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]

        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )

        # Call backend for forward pass
        result = await backend.forward_async(
            prompt=formatted_prompt,
            max_length=2048,
            **kwargs
        )

        return result

    def get_assistant_prefix(self) -> str:
        """Get the ChatML assistant prefix used by Qwen3 models."""
        return "<|im_start|>assistant\n"

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_preference(self, raw_output: str) -> int:
        """
        Parse preference from model output.

        Looks for \\boxed{1} or \\boxed{2} in the output.

        Args:
            raw_output: Raw model output

        Returns:
            1 or 2 indicating preference

        Raises:
            ValueError: If preference cannot be parsed
        """
        # Look for \\boxed{1} or \\boxed{2}
        match = re.search(r'\\boxed\{(\d+)\}', raw_output)
        if match:
            preference = int(match.group(1))
            if preference in [1, 2]:
                return preference

        # Fallback: look for just "1" or "2" at end of output
        if raw_output.strip().endswith("1"):
            return 1
        elif raw_output.strip().endswith("2"):
            return 2

        raise ValueError(f"Could not parse preference from output: {raw_output}")

    def _extract_reasoning(self, raw_output: str) -> Optional[str]:
        """
        Extract reasoning text from model output.

        Gets the text before the final \\boxed{} decision.

        Args:
            raw_output: Raw model output

        Returns:
            Reasoning text, or None if not found
        """
        # Find the \\boxed{} part
        match = re.search(r'\\boxed\{(\d+)\}', raw_output)
        if match:
            # Get text before the boxed part
            reasoning = raw_output[:match.start()].strip()
            if reasoning:
                return reasoning

        return None
