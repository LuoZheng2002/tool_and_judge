"""
Unified GPT-5 interface for both tool and judge projects.

This interface supports:
- Tool project: Function calling with OpenAI's structured outputs
- Judge project: Preference comparison (perplexity not available for API models)

Key features:
- Uses responses.create() API (new in GPT-5)
- Structured tool calling with optional strict mode
- Function name sanitization for API compatibility
- Reasoning mode support
"""

import json
import re
from typing import List, Dict, Any, Union, Optional
from .base import (
    JudgeModelInterface,
    ToolModelInterface,
    ModelBackend,
    ComparisonResult,
    ForwardResult,
)


class GPT5Interface(JudgeModelInterface, ToolModelInterface):
    """
    Unified interface for GPT-5 family models supporting both tool and judge use cases.

    Note: As an API model, GPT-5 does not support perplexity calculation
    (forward pass). Judge project methods are limited to preference comparison.
    """

    def __init__(self, model_variant: str = "gpt-5", use_strict_mode: bool = False):
        """
        Initialize the GPT-5 interface.

        Args:
            model_variant: Model variant ("gpt-5", "gpt-5-mini", or "gpt-5-nano")
            use_strict_mode: Whether to use strict mode for structured outputs
        """
        # Validate model variant
        valid_variants = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
        if model_variant not in valid_variants:
            raise ValueError(
                f"Invalid model variant: {model_variant}. Must be one of {valid_variants}"
            )

        self.model_variant = model_variant
        self.use_strict_mode = use_strict_mode

        # Name mappings for function name sanitization
        self.name_mapping = {}  # sanitized_name -> original_name
        self.reverse_mapping = {}  # original_name -> sanitized_name

    # =========================================================================
    # ModelInterface Methods
    # =========================================================================

    def get_model_name(self) -> str:
        """Get the model name/identifier."""
        return self.model_variant

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
        Generate tool/function calls from a user query using GPT-5 API.

        This method:
        1. Preprocesses functions (sanitization, schema fixing)
        2. Builds developer + user messages
        3. Calls OpenAI responses.create() API
        4. Returns raw output as JSON string

        Note: backend should be an OpenAI client with responses API.

        Args:
            backend: The backend (OpenAI client) to use for inference
            functions: List of available function definitions
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            max_new_tokens: Maximum number of tokens to generate (unused for GPT-5)
            temperature: Sampling temperature (unused for GPT-5)
            **kwargs: Additional model-specific parameters

        Returns:
            Raw model output as JSON string
        """
        # Preprocess functions (sanitization + schema fixes)
        tools = self._preprocess_and_convert_to_tools(functions, prompt_passing_in_english)

        # Build developer message with strong instructions
        developer_message = {
            "role": "developer",
            "content": (
                "You are an expert in composing functions. "
                "You are given a question and a set of possible functions. "
                "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
                "If none of the functions can be used, point it out. "
                "If the given question lacks the parameters required by the function, also point it out.\n\n"
                "You should ONLY return function calls in your response. "
                "You MUST NOT include any other text, explanations, or direct answers. "
                "If you decide to invoke any function(s), you MUST use the provided tools. "
                "Do NOT attempt to answer the question directly without using the available functions."
            )
        }

        input_messages = [
            developer_message,
            {"role": "user", "content": user_query}
        ]

        # Build API call parameters
        api_params = {
            "input": input_messages,
            "model": self.model_variant,
            "store": False,
        }

        # Add reasoning parameters for GPT-5
        if "gpt-5" in self.model_variant:
            api_params["reasoning"] = {"summary": "auto"}
            api_params["include"] = ["reasoning.encrypted_content"]

        # Add tools if provided
        if tools:
            api_params["tools"] = tools

        # Call the API through backend (which should be an OpenAI client)
        client = backend
        if not hasattr(client, 'responses'):
            raise TypeError(
                "Backend must be an OpenAI client with responses API. "
                "Got: " + str(type(client))
            )

        response = client.responses.create(**api_params)

        # Parse response
        model_responses = []

        for item in response.output:
            if item.type == "function_call":
                model_responses.append({
                    "type": "function_call",
                    "name": item.name,
                    "arguments": item.arguments,
                    "call_id": item.call_id
                })
            elif item.type == "reasoning":
                reasoning_text = ""
                if hasattr(item, 'summary') and item.summary:
                    for summary in item.summary:
                        reasoning_text += summary.text + "\n"
                model_responses.append({
                    "type": "reasoning",
                    "content": reasoning_text
                })

        # Return function calls or fallback to text
        if not any(r["type"] == "function_call" for r in model_responses):
            return json.dumps({
                "output_text": response.output_text,
                "items": model_responses
            })

        return json.dumps({"function_calls": model_responses})

    def preprocess_functions(
        self,
        functions: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Preprocess function definitions for GPT-5.

        This performs:
        - Function name sanitization (alphanumeric + underscore/hyphen only)
        - Schema fixing (dict->object, float->number)
        - Stores name mappings for postprocessing

        Args:
            functions: List of function definitions
            **kwargs: Additional parameters (e.g., prompt_passing_in_english)

        Returns:
            Preprocessed function definitions in GPT-5 tools format
        """
        prompt_passing_in_english = kwargs.get('prompt_passing_in_english', True)
        return self._preprocess_and_convert_to_tools(functions, prompt_passing_in_english)

    def postprocess_tool_calls(
        self,
        raw_output: str,
        **kwargs
    ) -> Union[List[Dict[str, Dict[str, Any]]], str]:
        """
        Postprocess raw output from GPT-5's structured response format.

        Args:
            raw_output: Raw JSON string output from the model
            **kwargs: Additional postprocessing parameters (e.g., name_mapper)

        Returns:
            List of function call dictionaries in format: [{func_name: {arguments}}, ...]
            Returns error string if parsing fails
        """
        try:
            # Parse the JSON response
            response_data = json.loads(raw_output)

            # Handle error responses
            if "error" in response_data:
                return f"Error from model: {response_data['error']}"

            # Check if we have function calls
            if "function_calls" in response_data:
                function_calls = response_data["function_calls"]

                # Convert function calls to standard format
                extracted = []
                for func_call in function_calls:
                    if func_call.get("type") == "function_call":
                        sanitized_name = func_call.get("name")
                        arguments = func_call.get("arguments", {})

                        # Parse arguments if they come as a JSON string
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                pass

                        if sanitized_name:
                            # Convert sanitized name back to original
                            name_mapper = kwargs.get('name_mapper')
                            if name_mapper:
                                original_name = name_mapper.get_original_name(sanitized_name)
                            else:
                                original_name = self.name_mapping.get(sanitized_name, sanitized_name)

                            # Convert to standard format
                            extracted.append({original_name: arguments})

                if extracted:
                    return extracted
                else:
                    return "No function calls found in response"

            # Fallback: no function calls
            elif "output_text" in response_data:
                return f"Model returned text instead of function calls: {str(response_data['output_text'])[:200]}..."
            else:
                return f"Unexpected response format: {json.dumps(response_data)[:200]}..."

        except json.JSONDecodeError as e:
            return f"Failed to parse JSON output: {str(e)}. Raw string: {raw_output}"
        except Exception as e:
            return f"Error parsing output: {str(e)}. Raw string: {raw_output}"

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
        2. Calls OpenAI API to generate
        3. Parses output to extract preference (1 or 2)

        Args:
            backend: The backend (OpenAI client) to use for inference
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
            {
                "role": "developer",
                "content": "You are an expert judge. Provide only the final decision in the requested format."
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ]

        # Call API
        client = backend
        response = client.responses.create(
            input=messages,
            model=self.model_variant,
            store=False
        )

        # Parse preference from output
        raw_output = response.output_text
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
        2. Calls OpenAI API with reasoning enabled
        3. Parses output to extract both reasoning and preference

        Args:
            backend: The backend (OpenAI client) to use for inference
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
            {
                "role": "developer",
                "content": "You are an expert judge. Think through your reasoning before providing the final decision."
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ]

        # Call API with reasoning enabled
        client = backend
        response = client.responses.create(
            input=messages,
            model=self.model_variant,
            store=False,
            reasoning={"summary": "auto"},
            include=["reasoning.encrypted_content"]
        )

        # Extract reasoning from response
        reasoning_text = ""
        for item in response.output:
            if item.type == "reasoning" and hasattr(item, 'summary') and item.summary:
                for summary in item.summary:
                    reasoning_text += summary.text + "\n"

        # Parse preference from output
        raw_output = response.output_text
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
            "Perplexity calculation (forward_for_logits) is not supported for API models like GPT-5. "
            "This requires direct access to model logits which is not available through the API."
        )

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
        # Find the \\boxed{} part
        match = re.search(r'\\boxed\{(\d+)\}', raw_output)
        if match:
            # Get text before the boxed part
            reasoning = raw_output[:match.start()].strip()
            if reasoning:
                return reasoning

        return None

    def _sanitize_function_name(self, name: str, existing_sanitized: set) -> str:
        """
        Sanitize function name to match GPT-5's requirements.

        GPT-5 requires function names to match: ^[a-zA-Z0-9_-]+$

        Args:
            name: Original function name
            existing_sanitized: Set of already-used sanitized names

        Returns:
            Sanitized function name
        """
        # Replace dots with underscores
        sanitized = name.replace(".", "_")
        # Replace any other invalid characters
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', sanitized)

        # Handle collisions
        if sanitized in existing_sanitized:
            counter = 1
            base_sanitized = sanitized
            while f"{base_sanitized}_{counter}" in existing_sanitized:
                counter += 1
            sanitized = f"{base_sanitized}_{counter}"

        return sanitized

    def _fix_schema_basic(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply basic schema fixes for non-strict mode.

        Args:
            schema: Original JSON schema

        Returns:
            Schema with basic fixes applied
        """
        if not isinstance(schema, dict):
            return schema

        fixed = {}

        for key, value in schema.items():
            if key == "type":
                # Fix invalid types
                if value == "dict":
                    fixed[key] = "object"
                elif value == "float":
                    fixed[key] = "number"
                elif value == "tuple":
                    fixed[key] = "array"
                elif value == "any":
                    continue  # Skip "any" type
                else:
                    fixed[key] = value
            elif isinstance(value, dict):
                fixed[key] = self._fix_schema_basic(value)
            elif isinstance(value, list):
                fixed[key] = [
                    self._fix_schema_basic(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                fixed[key] = value

        return fixed

    def _fix_schema_for_strict_mode(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix JSON schema for GPT-5's strict mode.

        Args:
            schema: Original JSON schema

        Returns:
            Schema compatible with strict mode
        """
        if not isinstance(schema, dict):
            return schema

        fixed = {}

        for key, value in schema.items():
            if key == "type":
                # Fix invalid types
                if value == "dict":
                    fixed[key] = "object"
                elif value == "float":
                    fixed[key] = "number"
                elif value == "tuple":
                    fixed[key] = "array"
                elif value == "any":
                    continue
                else:
                    fixed[key] = value
            elif key == "properties" and isinstance(value, dict):
                # Recursively fix nested schemas
                fixed[key] = {
                    prop_name: self._fix_schema_for_strict_mode(prop_value)
                    for prop_name, prop_value in value.items()
                }
                # Make all properties required in strict mode
                all_properties = set(value.keys())
                fixed["required"] = sorted(all_properties)
            elif key == "required":
                # Skip - handled when processing properties
                if "properties" not in schema:
                    fixed[key] = value if isinstance(value, list) else [value]
            elif isinstance(value, dict):
                fixed[key] = self._fix_schema_for_strict_mode(value)
            elif isinstance(value, list):
                fixed[key] = [
                    self._fix_schema_for_strict_mode(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                fixed[key] = value

        # Add additionalProperties: false for objects in strict mode
        if fixed.get("type") == "object" and "properties" in fixed:
            if "additionalProperties" not in fixed:
                fixed["additionalProperties"] = False

        return fixed

    def _preprocess_and_convert_to_tools(
        self,
        functions: List[Dict[str, Any]],
        prompt_passing_in_english: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Preprocess and convert function definitions to GPT-5 tools format.

        This performs:
        - Function name sanitization
        - Schema fixing
        - Stores name mappings

        Args:
            functions: List of functions in BFCL format
            prompt_passing_in_english: Whether to add English parameter instruction

        Returns:
            List of tools in GPT-5 format
        """
        tools = []
        # Clear mappings
        self.name_mapping = {}
        self.reverse_mapping = {}
        existing_sanitized = set()

        for func in functions:
            original_name = func.get("name")
            func_description = func.get("description", "")
            func_parameters = func.get("parameters", {})

            # Sanitize function name
            sanitized_name = self._sanitize_function_name(original_name, existing_sanitized)
            existing_sanitized.add(sanitized_name)

            # Store mapping
            self.name_mapping[sanitized_name] = original_name
            self.reverse_mapping[original_name] = sanitized_name

            # Add English parameter instruction
            if prompt_passing_in_english and func_description:
                func_description = f"{func_description} (Pass parameters in English)"

            # Fix schema
            if self.use_strict_mode:
                fixed_parameters = self._fix_schema_for_strict_mode(func_parameters)
            else:
                fixed_parameters = self._fix_schema_basic(func_parameters)

            # Build tool
            tool = {
                "type": "function",
                "name": sanitized_name,
                "description": func_description,
                "parameters": fixed_parameters
            }

            # Add strict mode settings
            if self.use_strict_mode:
                tool["parameters"]["additionalProperties"] = False
                tool["strict"] = True

            tools.append(tool)

        return tools
