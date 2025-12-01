"""
Unified DeepSeek interface for both tool and judge projects.

This interface supports:
- Tool project: Function calling with Python syntax output
- Judge project: Preference comparison (perplexity not available for API models)

DeepSeek models:
- deepseek-chat: Standard chat model
- deepseek-reasoner: Model with reasoning capabilities
- Uses OpenAI-compatible API
- Outputs function calls in Python syntax: [func_name(param=value)]
"""

import ast
import json
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


class DeepSeekInterface(JudgeModelInterface, ToolModelInterface):
    """
    Unified interface for DeepSeek models supporting both tool and judge use cases.

    Note: As an API model, DeepSeek does not support perplexity calculation
    (forward pass). Judge project methods are limited to preference comparison.
    """

    def __init__(self, model_name: str = "deepseek-chat"):
        """
        Initialize the DeepSeek interface.

        Args:
            model_name: DeepSeek model variant (e.g., "deepseek-chat", "deepseek-reasoner")
        """
        self.model_name = model_name
        self.is_reasoner = "reasoner" in model_name.lower()

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
        Generate tool/function calls from a user query using DeepSeek API.

        This method:
        1. Builds system prompt with function definitions
        2. Calls OpenAI-compatible API
        3. Returns raw output (Python function call syntax)

        Note: backend should be an OpenAI-compatible client.

        Args:
            backend: The backend (OpenAI-compatible client) to use for inference
            functions: List of available function definitions
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            max_new_tokens: Maximum number of tokens to generate (unused for DeepSeek)
            temperature: Sampling temperature (default: 0.0)
            **kwargs: Additional model-specific parameters

        Returns:
            Raw model output as a string
        """
        # Build system prompt
        system_prompt = self._generate_system_prompt(
            functions=functions,
            prompt_passing_in_english=prompt_passing_in_english
        )

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        # Call API through backend (which should be an OpenAI-compatible client)
        client = backend
        if not hasattr(client, 'chat'):
            raise TypeError(
                "Backend must be an OpenAI-compatible client with chat.completions API. "
                "Got: " + str(type(client))
            )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )

        return response.choices[0].message.content

    def postprocess_tool_calls(
        self,
        raw_output: str,
        name_mapper: Optional['FunctionNameMapper'] = None
    ) -> Union[List[Dict[str, Dict[str, Any]]], str]:
        """
        Postprocess raw output from DeepSeek model to extract function calls.

        DeepSeek outputs in Python function call syntax:
        [func_name1(param1=value1, param2=value2), func_name2(param3=value3)]

        Parses using AST (Abstract Syntax Tree).

        Args:
            raw_output: Raw string output from the model
            name_mapper: Unused for DeepSeek (no name sanitization needed)

        Returns:
            List of function call dictionaries in format: [{func_name: {arguments}}, ...]
            Returns error string if parsing fails
        """
        # Strip backticks and whitespace
        raw_output = raw_output.strip("`\n ")

        # Add brackets if missing
        if not raw_output.startswith("["):
            raw_output = "[" + raw_output
        if not raw_output.endswith("]"):
            raw_output = raw_output + "]"

        # Remove wrapping quotes
        cleaned_input = raw_output.strip().strip("'")

        try:
            # Parse as Python AST
            parsed = ast.parse(cleaned_input, mode="eval")
        except SyntaxError:
            return f"Failed to decode AST: Invalid syntax. Raw string: {cleaned_input}"

        # Extract function calls from AST
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(self._resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                if not isinstance(elem, ast.Call):
                    return f"Failed to decode AST: Expected AST Call node, but got {type(elem)}. Raw string: {cleaned_input}"
                extracted.append(self._resolve_ast_call(elem))

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
        2. Calls DeepSeek API to generate
        3. Parses output to extract preference (1 or 2)

        Args:
            backend: The backend (OpenAI-compatible client) to use for inference
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

        messages = [
            {"role": "system", "content": "You are an expert judge. Provide only the final decision in the requested format."},
            {"role": "user", "content": prompt_text}
        ]

        # Call API
        client = backend
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
        )

        # Parse preference from output
        raw_output = response.choices[0].message.content
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

        For deepseek-reasoner, this enables the reasoning mode.

        Args:
            backend: The backend (OpenAI-compatible client) to use for inference
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

        messages = [
            {"role": "system", "content": "You are an expert judge. Think through your reasoning before providing the final decision."},
            {"role": "user", "content": prompt_text}
        ]

        # Call API
        client = backend
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
        )

        # Extract reasoning if available (deepseek-reasoner may include reasoning_content)
        raw_output = response.choices[0].message.content
        reasoning_text = None

        # Check if response has reasoning_content (for deepseek-reasoner)
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            reasoning_text = response.choices[0].message.reasoning_content

        # Parse preference from output
        preference = self._parse_preference(raw_output)

        # If no reasoning from API, extract from output text
        if not reasoning_text:
            reasoning_text = self._extract_reasoning(raw_output)

        return ComparisonResult(
            preference=preference,
            reasoning=reasoning_text if reasoning_text else None,
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
        Not supported for API models.

        Perplexity calculation requires access to model logits,
        which is not available through the API.

        Raises:
            NotImplementedError: Always (API models don't provide logits)
        """
        raise NotImplementedError(
            "Perplexity calculation (forward_for_logits) is not supported for API models like DeepSeek. "
            "This requires direct access to model logits which is not available through the API."
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_system_prompt(
        self,
        functions: List[Dict[str, Any]],
        prompt_passing_in_english: bool = True
    ) -> str:
        """
        Generate system prompt for DeepSeek based on available functions.

        Args:
            functions: List of available function definitions
            prompt_passing_in_english: Whether to request English parameter passing

        Returns:
            System prompt as a string
        """
        function_calls_json = json.dumps(functions, ensure_ascii=False, indent=2)
        passing_in_english_prompt = (
            " IMPORTANT: Pass in all parameters in function calls in English."
            if prompt_passing_in_english
            else ""
        )

        return f'''You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  You SHOULD NOT include any other text in the response.{passing_in_english_prompt}

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.
{function_calls_json}
'''

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
        import re
        # Look for \\boxed{1} or \\boxed{2}
        match = re.search(r'\\boxed\{(\d+)\}', raw_output)
        if match:
            preference = int(match.group(1))
            if preference in [1, 2]:
                return preference

        # Fallback: look for just "1" or "2" at end of output
        output_stripped = raw_output.strip()
        if output_stripped.endswith("1"):
            return 1
        elif output_stripped.endswith("2"):
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
        import re
        # Find the \\boxed{} part
        match = re.search(r'\\boxed\{(\d+)\}', raw_output)
        if match:
            # Get text before the boxed part
            reasoning = raw_output[:match.start()].strip()
            if reasoning:
                return reasoning

        return None

    def _resolve_ast_call(self, elem: ast.Call) -> Dict[str, Dict[str, Any]]:
        """
        Resolve an AST Call node to function call dictionary.

        Args:
            elem: AST Call node

        Returns:
            Dictionary in format: {func_name: {arguments}}
        """
        # Handle nested attributes for deeply nested module paths
        func_parts = []
        func_part = elem.func
        while isinstance(func_part, ast.Attribute):
            func_parts.append(func_part.attr)
            func_part = func_part.value
        if isinstance(func_part, ast.Name):
            func_parts.append(func_part.id)
        func_name = ".".join(reversed(func_parts))

        # Extract arguments
        args_dict = {}
        for arg in elem.keywords:
            output = self._resolve_ast_by_type(arg.value)
            args_dict[arg.arg] = output

        return {func_name: args_dict}

    def _resolve_ast_by_type(self, value: ast.expr) -> Any:
        """
        Resolve AST expression to Python value.

        Args:
            value: AST expression node

        Returns:
            Resolved Python value
        """
        if isinstance(value, ast.Constant):
            if value.value is Ellipsis:
                return "..."
            else:
                return value.value
        elif isinstance(value, ast.UnaryOp):
            return -value.operand.value
        elif isinstance(value, ast.List):
            return [self._resolve_ast_by_type(v) for v in value.elts]
        elif isinstance(value, ast.Dict):
            return {
                self._resolve_ast_by_type(k): self._resolve_ast_by_type(v)
                for k, v in zip(value.keys, value.values)
            }
        elif isinstance(value, ast.NameConstant):
            return value.value
        elif isinstance(value, ast.BinOp):
            return eval(ast.unparse(value))
        elif isinstance(value, ast.Name):
            # Convert lowercase "true" and "false" to Python's True and False
            if value.id == "true":
                return True
            elif value.id == "false":
                return False
            else:
                return value.id
        elif isinstance(value, ast.Call):
            if len(value.keywords) == 0:
                return ast.unparse(value)
            else:
                return self._resolve_ast_call(value)
        elif isinstance(value, ast.Tuple):
            # Convert tuple to list to match ground truth
            return [self._resolve_ast_by_type(v) for v in value.elts]
        elif isinstance(value, ast.Lambda):
            return eval(ast.unparse(value.body[0].value))
        elif isinstance(value, ast.Ellipsis):
            return "..."
        elif isinstance(value, ast.Subscript):
            try:
                return ast.unparse(value.body[0].value)
            except:
                return ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
        else:
            raise Exception(f"Unsupported AST type: {type(value)}")
