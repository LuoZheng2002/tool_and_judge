"""
Interface for Meta Llama 3.1 models via AWS Bedrock.

Handles:
- API calls to AWS Bedrock for Llama 3.1 (8B and 70B)
- Input formatting using Llama 3.1 chat template
- Output parsing from Python function call syntax
"""

import ast
import json
from typing import List, Dict, Any, Union
from models.base import ModelInterface


class Llama31Interface(ModelInterface):
    """Handler for Meta Llama 3.1 models via AWS Bedrock."""

    def __init__(self, model_id: str = "meta.llama3-1-8b-instruct-v1:0"):
        """
        Initialize the Llama 3.1 interface.

        Args:
            model_id: AWS Bedrock model ID for Llama 3.1
                     Options: "meta.llama3-1-8b-instruct-v1:0" or "meta.llama3-1-70b-instruct-v1:0"
        """
        self.model_id = model_id

    def infer(self, functions: List[Dict[str, Any]], user_query: str,
              prompt_passing_in_english: bool = True, model=None, generator=None) -> str:
        """
        Run inference with Llama 3.1 model via AWS Bedrock.

        Args:
            functions: List of available function definitions in JSON format
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            model: Unused for API models (kept for interface compatibility)
            generator: Bedrock client to use (required). This should be provided by the caller
                      and allows caching and reusing the client across multiple inferences.

        Returns:
            Raw model output as a string
        """
        # Use the provided client directly
        client = generator

        system_prompt = self._generate_system_prompt(
            functions=functions,
            prompt_passing_in_english=prompt_passing_in_english
        )

        # Format using Llama 3.1 chat template
        formatted_prompt = self._format_llama_3_1_chat_template(
            system_prompt=system_prompt,
            user_query=user_query
        )

        # Prepare request body for Bedrock
        request_body = {
            "prompt": formatted_prompt,
            "temperature": 0,
            "top_p": 0.9,
            "max_gen_len": 2048
        }

        # Call Bedrock API
        response = client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        # Parse response
        response_body = json.loads(response["body"].read())
        return response_body.get("generation", "")

    def parse_output(self, raw_output: str, name_mapper=None) -> Union[List[Dict[str, Dict[str, Any]]], str]:
        """
        Parse raw output from Llama 3.1 model using parse_ast.py strategy.

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
                               prompt_passing_in_english: bool = True) -> str:
        """
        Generate system prompt for Llama 3.1 model based on available functions.

        Adapted from main.py's gen_developer_prompt() function.

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

    def _format_llama_3_1_chat_template(self, system_prompt: str, user_query: str) -> str:
        """
        Format messages using the Llama 3.1 chat template.

        Llama 3.1 uses special tokens for chat formatting:
        <|begin_of_text|><|start_header_id|>role<|end_header_id|>
        content<|eot_id|>

        Args:
            system_prompt: System prompt as a string
            user_query: User query as a string

        Returns:
            Formatted prompt string using Llama 3.1's chat template
        """
        formatted_prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_query}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        return formatted_prompt

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
