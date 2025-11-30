"""
Interface for Anthropic Claude Sonnet model.

Handles:
- API calls to Anthropic
- Input formatting for Claude
- Output parsing from Python function call syntax
"""

import os
import ast
import json
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from models.base import ModelInterface


class ClaudeHaikuInterface(ModelInterface):
    """Handler for Anthropic Claude Haiku model."""

    def __init__(self):
        """Initialize the Claude Haiku interface."""
        load_dotenv(dotenv_path=".env")
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")

        # Lazy import to avoid dependency if not using this model
        from anthropic import Anthropic
        self.client = Anthropic(api_key=self.api_key)
        self.model_name = "claude-3-5-haiku-20241022"

    def infer(self, functions: List[Dict[str, Any]], user_query: str,
              prompt_passing_in_english: bool = True, model=None, generator=None) -> str:
        """
        Run inference with Claude Haiku.

        Args:
            functions: List of available function definitions in JSON format
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            model: Unused for API models (kept for interface compatibility)
            generator: Unused for API models (kept for interface compatibility)

        Returns:
            Raw model output as a string
        """
        system_prompt = self._generate_system_prompt(
            functions=functions,
            prompt_passing_in_english=prompt_passing_in_english,
            is_granite=False
        )

        # Claude expects messages without system in the messages list
        messages = [
            {"role": "user", "content": user_query}
        ]

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            temperature=0,
            system=system_prompt,
            messages=messages
        )

        # Claude returns structured content (list of message blocks)
        return response.content[0].text if response.content else ""

    def parse_output(self, raw_output: str) -> Union[List[Dict[str, Any]], str]:
        """
        Parse raw output from Claude Sonnet using parse_ast.py strategy.

        Expects Python function call syntax:
        [func_name1(param1=value1, param2=value2), func_name2(param3=value3)]

        Follows the same parsing strategy as parse_ast.py's raw_to_json() for API models.

        Args:
            raw_output: Raw string output from the model

        Returns:
            List of function call dictionaries in format: [{func_name: {arguments}}, ...]
            Returns error string if parsing fails (matching raw_to_json behavior)
        """
        # Strip backticks and whitespace (from parse_ast.py:170)
        raw_output = raw_output.strip("`\n ")

        # Add brackets if missing (from parse_ast.py:171-174)
        if not raw_output.startswith("["):
            raw_output = "[" + raw_output
        if not raw_output.endswith("]"):
            raw_output = raw_output + "]"

        # Remove wrapping quotes (from parse_ast.py:176)
        cleaned_input = raw_output.strip().strip("'")

        try:
            # Parse as Python AST (from parse_ast.py:178)
            parsed = ast.parse(cleaned_input, mode="eval")
        except SyntaxError:
            return f"Failed to decode AST: Invalid syntax. Raw string: {cleaned_input}"

        # Extract function calls from AST (from parse_ast.py:181-189)
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(self._resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                if not isinstance(elem, ast.Call):
                    return f"Failed to decode AST: Expected AST Call node, but got {type(elem)}. Raw string: {cleaned_input}"
                extracted.append(self._resolve_ast_call(elem))

        return extracted

    def _generate_system_prompt(self, functions: List[Dict[str, Any]],
                               prompt_passing_in_english: bool = True,
                               is_granite: bool = False) -> str:
        """
        Generate system prompt for the model based on available functions.

        Adapted from main.py's gen_developer_prompt() function.

        Args:
            functions: List of available function definitions
            prompt_passing_in_english: Whether to request English parameter passing
            is_granite: Whether this is for Granite model (uses different format)

        Returns:
            System prompt as a string
        """
        function_calls_json = json.dumps(functions, ensure_ascii=False, indent=2)
        passing_in_english_prompt = (
            " IMPORTANT: Pass in all parameters in function calls in English."
            if prompt_passing_in_english
            else ""
        )

        if is_granite:
            # Granite format - JSON output
            return f'''You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your response, in JSON format as a list where each element has the format {{"name": "function_name", "arguments": {{param1: value1, param2: value2, ...}}}}.{passing_in_english_prompt}

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user\'s request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.
{function_calls_json}
'''
        else:
            # API format - Python function call syntax
            return f'''You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  You SHOULD NOT include any other text in the response.{passing_in_english_prompt}

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user\"s request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.
{function_calls_json}
'''

    def _resolve_ast_call(self, elem: ast.Call) -> Dict[str, Dict[str, Any]]:
        """
        Resolve an AST Call node to function call dictionary.

        This is adapted from parse_ast.py's resolve_ast_call function.

        Args:
            elem: AST Call node

        Returns:
            Dictionary in format: {func_name: {arguments}}
        """
        # Handle nested attributes for deeply nested module paths (from parse_ast.py:111-119)
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

        This is adapted from parse_ast.py's resolve_ast_by_type function.

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
            # Convert tuple to list to match ground truth (from parse_ast.py:96)
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
