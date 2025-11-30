"""
Interface for Alibaba Qwen2.5 7B Instruct local model.

Handles:
- Local model inference via generator pipeline
- Qwen2.5 chat template formatting with special tokens
- Output parsing from JSON format (tool_call)
"""

import json
from typing import List, Dict, Any, Union
from models.base import ModelInterface

try:
    from config import LocalModel
except ImportError:
    LocalModel = None


class Qwen25InstructInterface(ModelInterface):
    """Handler for Alibaba Qwen2.5 7B Instruct local model."""

    def __init__(self, generator, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        Initialize the Qwen2.5 interface.

        Args:
            generator: Optional pre-initialized generator from pipeline
                      If None, you must provide it via set_generator() before calling infer()
            model_id: Model identifier string
                     Options: "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
                             "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-72B-Instruct"
        """
        self.generator = generator
        self.model_id = model_id

    def infer(self, functions: List[Dict[str, Any]], user_query: str,
              prompt_passing_in_english: bool = True, model=None) -> str:
        """
        Run inference with Qwen2.5 model.

        Args:
            functions: List of available function definitions in JSON format
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            model: Should be LocalModel.QWEN_2_5_7B_INSTRUCT (for system prompt generation)

        Returns:
            Raw model output as a string
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized. Call set_generator() first.")

        system_prompt = self._generate_system_prompt(
            functions=functions,
            prompt_passing_in_english=prompt_passing_in_english
        )

        # Format the input using Qwen2.5 chat template with functions
        template = self._format_qwen2_5_chat_template(
            system_prompt=system_prompt,
            user_query=user_query,
            functions=functions,
            add_generation_prompt=True
        )

        # Call generate method on wrapper
        result = self.generator.generate(template)
        return result

    # Removed infer_batch() - now uses base class implementation with ThreadPoolExecutor
    # This allows concurrent request submission, and vLLM handles internal batching

    def parse_output(self, raw_output: str) -> Union[List[Dict[str, Dict[str, Any]]], str]:
        """
        Parse raw output from Qwen2.5 model using parse_ast.py strategy.

        Qwen2.5 outputs in JSON format within <tool_call> tags:
        <tool_call>
        {"name": "function_name", "arguments": {"param1": value1, ...}}
        </tool_call>

        Follows the same parsing strategy as parse_ast.py's raw_to_json() for Granite/Qwen models.

        Args:
            raw_output: Raw string output from the model

        Returns:
            List of function call dictionaries in format: [{func_name: {arguments}}, ...]
            Returns error string if parsing fails (matching raw_to_json behavior)
        """
        # Parse Qwen2.5 model's output format: <tool_call>{...}</tool_call>
        model_result_raw = raw_output.strip()

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

        # Convert Qwen2.5 format to desired format
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

    def _generate_system_prompt(self, functions: List[Dict[str, Any]],
                               prompt_passing_in_english: bool = True) -> str:
        """
        Generate system prompt for Qwen2.5 model based on available functions.

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

You should only return the function calls in your response, in JSON format as a list where each element has the format {{"name": "function_name", "arguments": {{param1: value1, param2: value2, ...}}}}.{passing_in_english_prompt}

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.
{function_calls_json}
'''

    def _format_qwen2_5_chat_template(self, system_prompt: str, user_query: str,
                                      functions: List[Dict[str, Any]] = None,
                                      add_generation_prompt: bool = True) -> str:
        """
        Format messages using the Qwen2.5 chat template.

        Qwen2.5 uses special tokens: <|im_start|> and <|im_end|>
        Tools section is included with function definitions.

        Args:
            system_prompt: System prompt as a string
            user_query: User query as a string
            functions: Optional list of function definitions for tool calling
            add_generation_prompt: Whether to add the generation prompt at the end

        Returns:
            Formatted prompt string using Qwen2.5's chat template
        """
        formatted_prompt = ""

        # Use provided system prompt with Qwen2.5 tags
        system_content = system_prompt if system_prompt else (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        )

        # Add system prompt with Qwen2.5 tokens
        formatted_prompt += f"<|im_start|>system\n{system_content}<|im_end|>\n"

        # Add tools section if functions are provided
        if functions:
            formatted_prompt += (
                "<|im_start|>system\n"
                "# Tools\n\n"
                "You may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                "<tools>\n"
            )
            for func in functions:
                formatted_prompt += json.dumps(func, ensure_ascii=False) + "\n"
            formatted_prompt += (
                "</tools>\n\n"
                "For each function call, return a json object with function name and arguments "
                "within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": <function-name>, "arguments": <args-json-object>}\n'
                "</tool_call><|im_end|>\n"
            )

        # Add user message
        formatted_prompt += f"<|im_start|>user\n{user_query}<|im_end|>\n"

        # Add generation prompt if requested
        if add_generation_prompt:
            formatted_prompt += "<|im_start|>assistant\n"

        return formatted_prompt
