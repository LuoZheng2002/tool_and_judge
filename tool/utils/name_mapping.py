"""
Model-agnostic function name sanitization and mapping utility.

Some models (like GPT-5) have restrictions on function names (e.g., no dots allowed).
This module provides a standalone utility to handle name sanitization and maintain
bidirectional mappings between original and sanitized names.
"""

import re
from typing import List, Dict, Any, Set


class FunctionNameMapper:
    """
    Model-agnostic function name sanitizer and mapper.

    Handles:
    - Sanitizing function names to meet API requirements (e.g., remove dots)
    - Maintaining bidirectional mappings between original and sanitized names
    - Collision detection and resolution
    """

    def __init__(self):
        """Initialize empty name mappings."""
        self.name_mapping: Dict[str, str] = {}  # sanitized -> original
        self.reverse_mapping: Dict[str, str] = {}  # original -> sanitized

    def populate_from_functions(self, functions: List[Dict[str, Any]]) -> None:
        """
        Build name mappings from function definitions.

        This method is model-agnostic and doesn't require any model to be loaded.
        It simply processes function metadata and builds the mapping tables.

        Args:
            functions: List of function definitions in BFCL format
        """
        # Clear existing mappings
        self.name_mapping = {}
        self.reverse_mapping = {}
        existing_sanitized: Set[str] = set()

        for func in functions:
            original_name = func.get("name")
            if original_name:
                # Sanitize the name and handle collisions
                sanitized_name = self._sanitize_name(original_name, existing_sanitized)
                existing_sanitized.add(sanitized_name)

                # Store bidirectional mappings
                self.name_mapping[sanitized_name] = original_name
                self.reverse_mapping[original_name] = sanitized_name

    def get_original_name(self, sanitized_name: str) -> str:
        """
        Convert sanitized name back to original name.

        Args:
            sanitized_name: Sanitized function name

        Returns:
            Original function name, or sanitized_name if no mapping exists
        """
        return self.name_mapping.get(sanitized_name, sanitized_name)

    def get_sanitized_name(self, original_name: str) -> str:
        """
        Get sanitized version of original name.

        Args:
            original_name: Original function name

        Returns:
            Sanitized function name, or original_name if no mapping exists
        """
        return self.reverse_mapping.get(original_name, original_name)

    def _sanitize_name(self, name: str, existing_sanitized: Set[str]) -> str:
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


# Global singleton instance for use across the application
_global_name_mapper = FunctionNameMapper()


def get_global_name_mapper() -> FunctionNameMapper:
    """
    Get the global FunctionNameMapper instance.

    This singleton is shared across all model interfaces that need name sanitization,
    avoiding the need to maintain separate mappings per model instance.

    Returns:
        The global FunctionNameMapper instance
    """
    return _global_name_mapper
