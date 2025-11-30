"""
Unified Qwen3 interface for both tool and judge projects.

This interface supports:
- Tool project: Function calling with tool_call format
- Judge project: Perplexity calculation and preference comparison

Qwen3 models use ChatML format with:
- Special tokens: <|im_start|> and <|im_end|>
- Thinking mode: <think></think> tags for chain-of-thought reasoning
- Tool calling: <tool_call>{...}</tool_call> format
"""

import json
from typing import List, Dict, Any, Union, Optional
from .base import JudgeModelInterface, ToolModelInterface, ModelBackend


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

    def build_prompt(
        self,
        user_query: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Build a basic formatted prompt for Qwen3.

        Args:
            user_query: User query string
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Formatted prompt string
        """
        formatted = ""

        # Add system message if provided
        if system_prompt:
            formatted += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        # Add user message
        formatted += f"<|im_start|>user\n{user_query}<|im_end|>\n"

        # Add generation prompt
        formatted += "<|im_start|>assistant\n"

        # Disable thinking by default unless enabled
        if not self.enable_thinking:
            formatted += "<think>\n\n</think>\n\n"

        return formatted

    # =========================================================================
    # JudgeModelInterface Methods
    # =========================================================================

    def get_system_message(self) -> str:
        """Get the default system message for Qwen3 models."""
        # Not used in current implementation
        raise NotImplementedError("System message is not used in current implementation.")

    def get_assistant_prefix(self) -> str:
        """Get the ChatML assistant prefix used by Qwen3 models."""
        return "<|im_start|>assistant\n"

    def build_messages_for_perplexity_forward(
        self,
        tokenizer: Any,
        question: str,
        answer: str,
        language: str
    ) -> str:
        """
        Build message structure for perplexity calculation (forward pass).

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            answer: The assistant's answer
            language: Formal language name (e.g., "Chinese", "English")

        Returns:
            Formatted conversation string
        """
        # Build language-specific instructions
        if language.lower() == "english":
            instruction = "Please answer the question in English with a concise phrase instead of a complete sentence. Start with an uncapitalized first word."
        else:
            instruction = f"Please answer the question in {language} with a concise phrase instead of a complete sentence."

        # Combine question with instruction
        user_content = f"{question}\n\n{instruction}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )

    def build_messages_for_perplexity_generate(
        self,
        tokenizer: Any,
        question: str,
        language: str
    ) -> str:
        """
        Build message structure for answer generation.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            language: Formal language name (e.g., "Chinese", "English")

        Returns:
            Formatted conversation string with generation prompt
        """
        # Build language-specific instructions
        if language.lower() == "english":
            instruction = "Please answer the question in English with a concise phrase instead of a complete sentence. Start with an uncapitalized first word."
        else:
            instruction = f"Please answer the question in {language} with a concise phrase instead of a complete sentence."

        # Combine question with instruction
        user_content = f"{question}\n\n{instruction}"

        messages = [
            {"role": "user", "content": user_content}
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

    def build_messages_for_compare_directly(
        self,
        tokenizer: Any,
        question: str,
        answer1: str,
        answer2: str
    ) -> str:
        """
        Build message structure for direct comparison.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            answer1: The first answer to compare
            answer2: The second answer to compare

        Returns:
            Formatted conversation string
        """
        prompt = f"""Given the following question and two answers, which answer is better?

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Provide your judgment IMMEDIATELY without reasoning or explanation. Provide your final decision in the following format:
\\boxed{{X}} where X is either 1 or 2."""

        messages = [
            {"role": "user", "content": prompt}
        ]

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        return formatted

    def build_messages_for_compare_cot(
        self,
        tokenizer: Any,
        question: str,
        answer1: str,
        answer2: str
    ) -> str:
        """
        Build message structure for comparison with chain-of-thought reasoning.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            answer1: The first answer to compare
            answer2: The second answer to compare

        Returns:
            Formatted conversation string
        """
        prompt = f"""Given the following question and two answers, which answer is better?

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Please briefly explain your reasoning, and then provide your final decision in the following format:
\\boxed{{X}} where X is either 1 or 2."""

        messages = [
            {"role": "user", "content": prompt}
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

    # =========================================================================
    # ToolModelInterface Methods
    # =========================================================================

    def infer_with_functions(
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
        Run inference with function definitions.

        Args:
            backend: The backend to use for inference
            functions: List of available function definitions
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Raw model output as a string
        """
        # Build prompt with functions
        prompt = self.build_prompt_with_functions(
            functions=functions,
            user_query=user_query,
            prompt_passing_in_english=prompt_passing_in_english,
            **kwargs
        )

        # Call backend
        result = backend.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )

        return result.generated_text

    def build_prompt_with_functions(
        self,
        functions: List[Dict[str, Any]],
        user_query: str,
        prompt_passing_in_english: bool = True,
        **kwargs
    ) -> str:
        """
        Build a formatted prompt with function definitions for Qwen3.

        Args:
            functions: List of function definitions
            user_query: User query string
            prompt_passing_in_english: Whether to request English parameter passing
            **kwargs: Additional parameters

        Returns:
            Formatted prompt string
        """
        # Generate system prompt
        system_prompt = self._generate_system_prompt(
            functions=functions,
            prompt_passing_in_english=prompt_passing_in_english
        )

        # Build full prompt with tools
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

        return formatted_prompt

    def parse_function_calls(
        self,
        raw_output: str,
        **kwargs
    ) -> Union[List[Dict[str, Dict[str, Any]]], str]:
        """
        Parse raw output from Qwen3 model to extract function calls.

        Qwen3 outputs in JSON format within <tool_call> tags:
        <tool_call>
        {"name": "function_name", "arguments": {"param1": value1, ...}}
        </tool_call>

        Args:
            raw_output: Raw string output from the model
            **kwargs: Additional parsing parameters

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
    # Helper Methods
    # =========================================================================

    def _generate_system_prompt(
        self,
        functions: List[Dict[str, Any]],
        prompt_passing_in_english: bool = True
    ) -> str:
        """
        Generate system prompt for Qwen3 model based on available functions.

        Args:
            functions: List of available function definitions
            prompt_passing_in_english: Whether to request English parameter passing

        Returns:
            System prompt as a string
        """
        passing_in_english_prompt = (
            " IMPORTANT: Pass in all parameters in function calls in English."
            if prompt_passing_in_english
            else ""
        )

        return f'''You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should ONLY return function calls in your response. You MUST NOT include any other text, explanations, or direct answers. If you decide to invoke any function(s), you MUST use the provided tools. Do NOT attempt to answer the question directly without using the available functions.{passing_in_english_prompt}

You should only return the function calls in your response, in JSON format as a list where each element has the format {{"name": "function_name", "arguments": {{param1: value1, param2: value2, ...}}}}.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.'''
