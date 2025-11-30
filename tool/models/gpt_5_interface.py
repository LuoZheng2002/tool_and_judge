"""
Interface for OpenAI GPT-5 family models.

Handles:
- API calls to OpenAI GPT-5, GPT-5-mini, and GPT-5-nano
- Input formatting using new responses API
- Tool calling with structured outputs
- Output parsing from GPT-5's structured response format

Key differences from GPT-4:
- Uses responses.create() instead of chat.completions.create()
- Uses 'input' parameter instead of 'messages'
- No system prompt - functions passed via 'tools' parameter
- Structured tool calling with strict mode
"""

import json
import re
from typing import List, Dict, Any, Union
from models.base import ModelInterface


class GPT5Interface(ModelInterface):
    """Handler for OpenAI GPT-5 family models."""

    def __init__(self, model_variant: str = "gpt-5", use_strict_mode: bool = False):
        """
        Initialize the GPT-5 interface.

        Args:
            model_variant: Model variant to use ("gpt-5", "gpt-5-mini", or "gpt-5-nano")
            use_strict_mode: Whether to use strict mode for structured outputs (default: False)
                           Strict mode requires all parameters to be required and has stricter
                           schema validation. Disable if you have optional parameters.
        """
        # Validate and set model variant
        valid_variants = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
        if model_variant not in valid_variants:
            raise ValueError(f"Invalid model variant: {model_variant}. Must be one of {valid_variants}")
        self.model_name = model_variant
        self.use_strict_mode = use_strict_mode

        # Mapping between sanitized names (for GPT-5 API) and original names (for output)
        self.name_mapping = {}  # sanitized_name -> original_name
        self.reverse_mapping = {}  # original_name -> sanitized_name

    def populate_name_mapping(self, functions: List[Dict[str, Any]]) -> None:
        """
        Populate name_mapping from function definitions without calling the API.

        This is useful when parse_output needs to be called without a prior infer() call
        (e.g., when parsing previously saved raw outputs).

        Args:
            functions: List of available function definitions in BFCL format
        """
        # Clear existing mappings
        self.name_mapping = {}
        self.reverse_mapping = {}
        existing_sanitized = set()

        for func in functions:
            original_name = func.get("name")
            if original_name:
                # Sanitize the name using the same logic as in _convert_functions_to_tools
                sanitized_name = self._sanitize_function_name(original_name, existing_sanitized)
                existing_sanitized.add(sanitized_name)

                # Store mappings
                self.name_mapping[sanitized_name] = original_name
                self.reverse_mapping[original_name] = sanitized_name

    def infer(self, functions: List[Dict[str, Any]], user_query: str,
              prompt_passing_in_english: bool = True, model=None, generator=None) -> str:
        """
        Run inference with GPT-5.

        Args:
            functions: List of available function definitions in BFCL format
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            model: Unused for API models (kept for interface compatibility)
            generator: OpenAI client to use (required). This should be provided by the caller
                      and allows caching and reusing the client across multiple inferences.

        Returns:
            Raw model output as JSON string
        """
        # Use the provided client directly
        client = generator

        # Convert BFCL functions to GPT-5 tools format
        tools = self._convert_functions_to_tools(functions, prompt_passing_in_english)

        # GPT-5 uses 'input' instead of 'messages'
        # Add a developer message with strong instructions to ONLY use functions
        # This prevents GPT-5 from answering directly instead of using tools
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
        kwargs = {
            "input": input_messages,
            "model": self.model_name,
            "store": False,
        }

        # GPT-5 is a reasoning model, so add reasoning parameters
        if "gpt-5" in self.model_name:
            kwargs["reasoning"] = {"summary": "auto"}
            kwargs["include"] = ["reasoning.encrypted_content"]

        # Add tools if provided
        if tools and len(tools) > 0:
            kwargs["tools"] = tools

        response = client.responses.create(**kwargs)

        # Parse response following BFCL's approach (openai_response.py:145-172)
        # The response.output is a list of items, each with a type field
        model_responses = []

        for item in response.output:
            if item.type == "function_call":
                # This is a function call - extract name and arguments
                model_responses.append({
                    "type": "function_call",
                    "name": item.name,
                    "arguments": item.arguments,
                    "call_id": item.call_id
                })
            elif item.type == "reasoning":
                # This is reasoning content
                reasoning_text = ""
                if hasattr(item, 'summary') and item.summary:
                    for summary in item.summary:
                        reasoning_text += summary.text + "\n"
                model_responses.append({
                    "type": "reasoning",
                    "content": reasoning_text
                })

        # If no function calls found, use output_text as fallback
        if not any(r["type"] == "function_call" for r in model_responses):
            return json.dumps({
                "output_text": response.output_text,
                "items": model_responses
            })

        return json.dumps({"function_calls": model_responses})

    def parse_output(self, raw_output: str, name_mapper=None) -> Union[List[Dict[str, Any]], str]:
        """
        Parse raw output from GPT-5's structured response format.

        Following BFCL's approach (openai_response.py:145-172), the response
        contains function_calls with name, arguments, and call_id.

        Args:
            raw_output: Raw JSON string output from the model
            name_mapper: Optional external FunctionNameMapper for name conversion.
                        If provided, uses this instead of self.name_mapping.
                        This allows parsing without loading the model.

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

            # Check if we have function calls in the new format
            if "function_calls" in response_data:
                function_calls = response_data["function_calls"]

                # Convert function calls to BFCL format
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
                                # If parsing fails, keep as string (will be caught later)
                                pass

                        if sanitized_name:
                            # Convert sanitized name back to original name
                            # Use external name_mapper if provided, otherwise fall back to self.name_mapping
                            if name_mapper:
                                original_name = name_mapper.get_original_name(sanitized_name)
                            else:
                                original_name = self.name_mapping.get(sanitized_name, sanitized_name)

                            # Convert to BFCL format: {func_name: {arguments}}
                            extracted.append({original_name: arguments})

                if extracted:
                    return extracted
                else:
                    return "No function calls found in response"

            # Fallback: no function calls, check for text output
            elif "output_text" in response_data:
                return f"Model returned text instead of function calls: {str(response_data['output_text'])[:200]}..."

            else:
                return f"Unexpected response format: {json.dumps(response_data)[:200]}..."

        except json.JSONDecodeError as e:
            return f"Failed to parse JSON output: {str(e)}. Raw string: {raw_output}"
        except Exception as e:
            return f"Error parsing output: {str(e)}. Raw string: {raw_output}"

    def _fix_schema_basic(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply basic schema fixes for non-strict mode.

        Only fixes critical type issues that are invalid in standard JSON Schema:
        - Replace type: "dict" with type: "object"
        - Replace type: "float" with type: "number"
        - Remove type: "any" (not a valid JSON Schema type)

        Does NOT enforce strict mode requirements (all properties required, etc.)

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
                    # Skip "any" type - omit the type field to allow any value
                    continue
                else:
                    fixed[key] = value
            elif isinstance(value, dict):
                # Recursively fix nested schemas
                fixed[key] = self._fix_schema_basic(value)
            elif isinstance(value, list):
                # Recursively fix schemas in lists
                fixed[key] = [
                    self._fix_schema_basic(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                fixed[key] = value

        return fixed

    def _fix_schema_for_strict_mode(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix JSON schema to be compatible with GPT-5's strict mode.

        GPT-5 strict mode requirements:
        - Replace type: "dict" with type: "object"
        - Replace type: "float" with type: "number"
        - Remove type: "any" (not a valid JSON Schema type)
        - All properties must be in the "required" array
        - All nested objects must have "additionalProperties": false
        - No unsupported JSON Schema features

        Args:
            schema: Original JSON schema (may be BFCL format)

        Returns:
            Fixed schema compatible with GPT-5 strict mode
        """
        if not isinstance(schema, dict):
            return schema

        # Create a deep copy to avoid modifying the original
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
                    # Skip "any" type - omit the type field to allow any value
                    continue
                else:
                    fixed[key] = value
            elif key == "properties" and isinstance(value, dict):
                # Recursively fix nested schemas
                fixed[key] = {
                    prop_name: self._fix_schema_for_strict_mode(prop_value)
                    for prop_name, prop_value in value.items()
                }
                # GPT-5 strict mode requires ALL properties to be in the "required" array
                # This is different from standard JSON Schema where properties can be optional
                # We merge existing required fields with all properties
                existing_required = schema.get("required", [])
                all_properties = set(value.keys())
                fixed["required"] = sorted(all_properties)  # Ensure all properties are required
            elif key == "required":
                # Skip - already handled when processing "properties"
                # If there are no properties, just copy the required array as-is
                if "properties" not in schema:
                    fixed[key] = value if isinstance(value, list) else [value]
            elif isinstance(value, dict):
                # Recursively fix nested schemas
                fixed[key] = self._fix_schema_for_strict_mode(value)
            elif isinstance(value, list):
                # Recursively fix schemas in lists
                fixed[key] = [
                    self._fix_schema_for_strict_mode(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                fixed[key] = value

        # Ensure nested objects have additionalProperties: false
        if fixed.get("type") == "object" and "properties" in fixed:
            if "additionalProperties" not in fixed:
                fixed["additionalProperties"] = False

        return fixed

    def _sanitize_function_name(self, name: str, existing_sanitized: set) -> str:
        """
        Sanitize function name to match GPT-5's requirements.

        GPT-5 requires function names to match pattern: ^[a-zA-Z0-9_-]+$
        (only letters, numbers, underscores, and hyphens)

        Handles collisions by appending a counter (_1, _2, etc.) if the sanitized
        name already exists.

        Args:
            name: Original function name (may contain dots, etc.)
            existing_sanitized: Set of already-used sanitized names to avoid collisions

        Returns:
            Sanitized function name safe for GPT-5 API (unique within this set)
        """
        # Replace dots with underscores (common in BFCL for nested functions)
        sanitized = name.replace(".", "_")
        # Replace any other invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', sanitized)

        # Handle collisions by appending a counter
        if sanitized in existing_sanitized:
            counter = 1
            base_sanitized = sanitized
            while f"{base_sanitized}_{counter}" in existing_sanitized:
                counter += 1
            sanitized = f"{base_sanitized}_{counter}"

        return sanitized

    def _convert_functions_to_tools(self, functions: List[Dict[str, Any]],
                                    prompt_passing_in_english: bool = True) -> List[Dict[str, Any]]:
        """
        Convert BFCL function format to GPT-5 tools format.

        BFCL format:
        {
            "name": "function_name",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

        GPT-5 tools format:
        {
            "type": "function",
            "name": "function_name",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...],
                "additionalProperties": False
            },
            "strict": True
        }

        Args:
            functions: List of functions in BFCL format
            prompt_passing_in_english: Whether to add English parameter instruction

        Returns:
            List of tools in GPT-5 format
        """
        tools = []
        # Clear mappings for this request
        self.name_mapping = {}
        self.reverse_mapping = {}
        existing_sanitized = set()

        for func in functions:
            # Extract function details
            original_name = func.get("name")
            func_description = func.get("description", "")
            func_parameters = func.get("parameters", {})

            # Sanitize function name for GPT-5 API compatibility
            # Pass existing sanitized names to avoid collisions
            sanitized_name = self._sanitize_function_name(original_name, existing_sanitized)
            existing_sanitized.add(sanitized_name)

            # Store mapping for later conversion back
            self.name_mapping[sanitized_name] = original_name
            self.reverse_mapping[original_name] = sanitized_name

            # Add instruction about English parameters to description if requested
            if prompt_passing_in_english and func_description:
                func_description = f"{func_description} (Pass parameters in English)"

            # Fix schema based on strict mode setting
            if self.use_strict_mode:
                # Fix schema to be compatible with GPT-5 strict mode
                fixed_parameters = self._fix_schema_for_strict_mode(func_parameters)
            else:
                # Non-strict mode: only fix critical issues (dict/float types)
                fixed_parameters = self._fix_schema_basic(func_parameters)

            # Build GPT-5 tool format
            tool = {
                "type": "function",
                "name": sanitized_name,  # Use sanitized name
                "description": func_description,
                "parameters": fixed_parameters
            }

            # Add strict mode specific requirements
            if self.use_strict_mode:
                tool["parameters"]["additionalProperties"] = False
                tool["strict"] = True

            tools.append(tool)

        return tools
