"""
Interface for IBM Granite 4 local model.

Handles:
- Local model inference via generator pipeline
- Granite 4 chat template formatting with special tokens
- Output parsing from JSON format (tool_call)
"""

import json
from typing import List, Dict, Any, Union
from models.base import ModelInterface

try:
    from config import LocalModel
except ImportError:
    LocalModel = None


class Granite4Interface(ModelInterface):
    """Handler for IBM Granite 4 local model."""

    def __init__(self, model_id: str = "ibm-granite/granite-4-8b-instruct"):
        """
        Initialize the Granite 4 interface.

        Args:
            model_id: Model identifier string
                     Options: "ibm-granite/granite-4-8b-instruct", "ibm-granite/granite-4-20b-instruct", etc.
        """
        self.model_id = model_id

    def infer(self, functions: List[Dict[str, Any]], user_query: str,
              prompt_passing_in_english: bool = True, model=None, generator=None) -> str:
        """
        Run inference with Granite 4 model.

        Args:
            functions: List of available function definitions in JSON format
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            model: Should be LocalModel.GRANITE_4 (for system prompt generation)
            generator: Generator instance for inference (required)

        Returns:
            Raw model output as a string
        """
        if generator is None:
            raise RuntimeError("Generator must be provided for local model inference.")

        system_prompt = self._generate_system_prompt(
            functions=functions,
            prompt_passing_in_english=prompt_passing_in_english
        )

        # Format the input using Granite 4 chat template with functions
        template = self._format_granite4_chat_template(
            system_prompt=system_prompt,
            user_query=user_query,
            functions=functions,
            add_generation_prompt=True
        )

        # Call generate method on wrapper
        result = generator.generate(template)
        return result

    # Removed infer_batch() - now uses base class implementation with ThreadPoolExecutor
    # This allows concurrent request submission, and vLLM handles internal batching

    def parse_output(self, raw_output: str, name_mapper=None) -> Union[List[Dict[str, Dict[str, Any]]], str]:
        """
        Parse raw output from Granite 4 model using parse_ast.py strategy.

        Granite 4 outputs in JSON format within <tool_call> tags:
        <tool_call>
        {"name": "function_name", "arguments": {"param1": value1, ...}}
        </tool_call>

        Follows the same parsing strategy as parse_ast.py's raw_to_json() for Granite models.

        Args:
            raw_output: Raw string output from the model

        Returns:
            List of function call dictionaries in format: [{func_name: {arguments}}, ...]
            Returns error string if parsing fails (matching raw_to_json behavior)
        """
        # Parse Granite 4 model's output format: <tool_call>{...}</tool_call>
        model_result_raw = raw_output.strip()

        # Extract content from <tool_call> tags if present
        if "<tool_call>" in model_result_raw:
            # Handle potential multiple tool calls
            tool_calls_list = []
            remaining = model_result_raw

            while "<tool_call>" in remaining:
                start_idx = remaining.find("<tool_call>")
                end_idx = remaining.find("</tool_call>")

                if start_idx != -1 and end_idx != -1:
                    tool_call_content = remaining[start_idx + len("<tool_call>"):end_idx].strip()
                    if tool_call_content:
                        tool_calls_list.append(tool_call_content)
                    remaining = remaining[end_idx + len("</tool_call>"):]
                else:
                    break

            # If we found tool calls, join them as a JSON array
            if tool_calls_list:
                # Each tool call should be a JSON object
                model_result_raw = "[" + ",".join(tool_calls_list) + "]"
            else:
                # Fallback: extract first tool call content
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

        # Convert Granite 4 format to desired format
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
        Generate system prompt for Granite 4 model based on available functions.

        Adapted from GPT-5's prompt structure with similar strong instructions
        to only use functions and not answer directly.

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

    def _format_granite4_chat_template(self, system_prompt: str, user_query: str,
                                       functions: List[Dict[str, Any]] = None,
                                       add_generation_prompt: bool = True) -> str:
        """
        Format messages using the Granite 4 chat template.

        Granite 4 uses special tokens: <|start_of_role|>, <|end_of_role|>, and <|end_of_text|>
        Tools section is included within the system message following the template format.

        Based on the granite_4_chat_template.txt template structure.

        Args:
            system_prompt: System prompt as a string
            user_query: User query as a string
            functions: Optional list of function definitions for tool calling
            add_generation_prompt: Whether to add the generation prompt at the end

        Returns:
            Formatted prompt string using Granite 4's chat template
        """
        formatted_prompt = ""

        # Build system message with tools if functions are provided
        if functions:
            # Following granite_4_chat_template.txt lines 1-2, build tools system message
            tools_system_message = (
                "You are a helpful assistant with access to the following tools. "
                "You may call one or more tools to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                "<tools>"
            )
            for func in functions:
                tools_system_message += "\n" + json.dumps(func, ensure_ascii=False)
            tools_system_message += (
                "\n</tools>\n\n"
                "For each tool call, return a json object with function name and arguments "
                "within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": <function-name>, "arguments": <args-json-object>}\n'
                "</tool_call>. If a tool does not exist in the provided list of tools, "
                "notify the user that you do not have the ability to fulfill the request."
            )

            # Combine system prompt with tools message (following template lines 43-46)
            full_system_message = system_prompt + "\n\n" + tools_system_message
        else:
            # Just use the system prompt or default message
            full_system_message = system_prompt if system_prompt else (
                "You are a helpful assistant. Please ensure responses are professional, accurate, and safe."
            )

        # Add system message with Granite 4 tags (following template lines 59-62)
        formatted_prompt += (
            f"<|start_of_role|>system<|end_of_role|>{full_system_message}<|end_of_text|>\n"
        )

        # Add user message (following template lines 80-81)
        formatted_prompt += (
            f"<|start_of_role|>user<|end_of_role|>{user_query}<|end_of_text|>\n"
        )

        # Add generation prompt if requested (following template lines 116-118)
        if add_generation_prompt:
            formatted_prompt += "<|start_of_role|>assistant<|end_of_role|>"

        return formatted_prompt
