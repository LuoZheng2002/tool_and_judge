"""
Abstract base classes defining the interface for all model handlers.

Each model-specific interface should inherit from ModelInterface and implement:
1. infer(): Takes system prompt and user query, returns raw model output
2. parse_output(): Takes raw output, returns parsed function calls
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


class ModelInterface(ABC):
    """
    Abstract base class for all model handlers.

    Each model implementation should:
    1. Accept system_prompt and user_query as strings
    2. Return raw model output as string from infer()
    3. Parse raw output to standardized function call list format
    """

    @abstractmethod
    def infer(self, functions: List[Dict[str, Any]], user_query: str,
              prompt_passing_in_english: bool = True, model=None, generator=None) -> str:
        """
        Run inference with the model.

        Args:
            functions: List of available function definitions in JSON format
            user_query: User query/question as a string
            prompt_passing_in_english: Whether to request English parameter passing (default: True)
            model: Optional model type for customizing system prompt (LocalModel enum for Granite)
            generator: Optional generator for local model inference (required for local models)

        Returns:
            Raw model output as a string
        """
        pass

    def infer_batch(self, functions_list: List[List[Dict[str, Any]]],
                    user_queries: List[str],
                    prompt_passing_in_english: bool = True,
                    generator=None) -> List[str]:
        """
        Run batch inference with the model using concurrent requests.

        For API models, this uses ThreadPoolExecutor to make parallel API calls.
        For local models like Granite with native batch support, override this method.

        Args:
            functions_list: List of function definition lists (one per query)
            user_queries: List of user queries as strings
            prompt_passing_in_english: Whether to request English parameter passing (default: True)
            generator: Optional generator for local model inference (required for local models)

        Returns:
            List of raw model outputs as strings (same order as input)
        """
        if len(functions_list) != len(user_queries):
            raise ValueError("functions_list and user_queries must have same length")

        results = [None] * len(user_queries)  # Pre-allocate to maintain order

        def call_infer_with_index(index_and_args):
            """Helper to call infer and track original index"""
            index, functions, user_query = index_and_args
            try:
                response = self.infer(
                    functions=functions,
                    user_query=user_query,
                    prompt_passing_in_english=prompt_passing_in_english,
                    generator=generator
                )
                return index, response
            except Exception as e:
                print(f"Error calling model for batch item {index}: {e}")
                return index, f"Error: {str(e)}"

        # Use ThreadPoolExecutor for concurrent calls
        max_workers = min(8, len(user_queries))  # Up to 8 concurrent requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(call_infer_with_index, (i, functions_list[i], user_queries[i])): i
                for i in range(len(user_queries))
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    index, response = future.result()
                    results[index] = response
                except Exception as e:
                    index = futures[future]
                    print(f"Error processing batch item {index}: {e}")
                    results[index] = f"Error: {str(e)}"

        return results

    @abstractmethod
    def parse_output(self, raw_output: str) -> List[Dict[str, Dict[str, Any]]]:
        """
        Parse raw model output to standardized function call format.

        This method follows the parsing strategy from parse_ast.py's raw_to_json() function.

        Args:
            raw_output: Raw string output from the model

        Returns:
            List of function call dictionaries in format:
            [
                {
                    "function_name": {
                        "param1": value1,
                        "param2": value2,
                        ...
                    }
                },
                ...
            ]

            For error cases, returns a string describing the error (same as raw_to_json()).

        Raises:
            ValueError: If output cannot be parsed (alternative to returning error string)
        """
        pass
