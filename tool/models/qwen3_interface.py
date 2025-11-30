"""
Interface for Alibaba Qwen3 local model.

Handles:
- Local model inference via generator pipeline
- Qwen3 chat template formatting with special tokens and reasoning support
- Output parsing from JSON format (tool_call)
- Reasoning capabilities with <think></think> tags
"""

import json
from typing import List, Dict, Any, Union
from models.base import ModelInterface

try:
    from config import LocalModel
except ImportError:
    LocalModel = None


class Qwen3Interface(ModelInterface):
    """Handler for Alibaba Qwen3 local model."""

    def __init__(self, model_id: str = "Qwen/Qwen3-8B-Instruct", enable_thinking: bool = False):
        """
        Initialize the Qwen3 interface.

        Args:
            model_id: Model identifier string
                     Options: "Qwen/Qwen3-8B-Instruct", "Qwen/Qwen3-14B-Instruct", etc.
            enable_thinking: Whether to enable Qwen3's reasoning mode (default: False)
                           When False, adds empty <think></think> tags to disable thinking
        """
        self.model_id = model_id
        self.enable_thinking = enable_thinking

    def infer(self, functions: List[Dict[str, Any]], user_query: str,
              prompt_passing_in_english: bool = True, model=None, generator=None) -> str:
        """
        Run inference with Qwen3 model.

        Args:
            functions: List of available function definitions in JSON format
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            model: Should be LocalModel.QWEN_3 (for system prompt generation)
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

        # Format the input using Qwen3 chat template with functions
        template = self._format_qwen3_chat_template(
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
        Parse raw output from Qwen3 model using parse_ast.py strategy.

        Qwen3 outputs in JSON format within <tool_call> tags:
        <tool_call>
        {"name": "function_name", "arguments": {"param1": value1, ...}}
        </tool_call>

        Qwen3 may also include reasoning content within <think></think> tags,
        which we ignore during parsing.

        Follows the same parsing strategy as parse_ast.py's raw_to_json() for Granite/Qwen models.

        Args:
            raw_output: Raw string output from the model

        Returns:
            List of function call dictionaries in format: [{func_name: {arguments}}, ...]
            Returns error string if parsing fails (matching raw_to_json behavior)
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

    def _generate_system_prompt(self, functions: List[Dict[str, Any]],
                               prompt_passing_in_english: bool = True) -> str:
        """
        Generate system prompt for Qwen3 model based on available functions.

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

    def _format_qwen3_chat_template(self, system_prompt: str, user_query: str,
                                    functions: List[Dict[str, Any]] = None,
                                    add_generation_prompt: bool = True) -> str:
        """
        Format messages using the Qwen3 chat template.

        Qwen3 uses special tokens: <|im_start|> and <|im_end|>
        Tools section is included with function definitions following the template format.
        Supports reasoning mode with <think></think> tags.

        Based on the qwen3_chat_template.txt template structure.

        Args:
            system_prompt: System prompt as a string
            user_query: User query as a string
            functions: Optional list of function definitions for tool calling
            add_generation_prompt: Whether to add the generation prompt at the end

        Returns:
            Formatted prompt string using Qwen3's chat template
        """
        formatted_prompt = ""

        # Add system prompt with tools if functions are provided
        if functions:
            # System message with tools section (following qwen3_chat_template.txt lines 2-11)
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
        else:
            # System message without tools (following qwen3_chat_template.txt lines 13-15)
            formatted_prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        # Add user message (following qwen3_chat_template.txt lines 31-32)
        formatted_prompt += f"<|im_start|>user\n{user_query}<|im_end|>\n"

        # Add generation prompt if requested (following qwen3_chat_template.txt lines 84-88)
        if add_generation_prompt:
            formatted_prompt += "<|im_start|>assistant\n"
            # Disable thinking mode by adding empty <think></think> tags if enable_thinking is False
            if not self.enable_thinking:
                formatted_prompt += "<think>\n\n</think>\n\n"

        return formatted_prompt
