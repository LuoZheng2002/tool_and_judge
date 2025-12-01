"""
Model-agnostic function name sanitization and mapping utility.

Some models (like GPT-5) have restrictions on function names (e.g., no dots allowed).
This module provides a standalone utility to handle name sanitization and maintain
bidirectional mappings between original and sanitized names.
"""

import re
from typing import Dict


class FunctionNameMapper:
    """
    Model-agnostic function name sanitizer and mapper with automatic caching.

    Handles:
    - Sanitizing function names to meet API requirements (e.g., remove dots)
    - Maintaining bidirectional mappings between original and sanitized names (as cache)
    - Collision detection and resolution

    The mappings act as a cache: get_sanitized_name() automatically sanitizes
    and caches new names on first access.
    """

    def __init__(self):
        """Initialize empty name mappings."""
        self.name_mapping: Dict[str, str] = {}  # sanitized -> original
        self.reverse_mapping: Dict[str, str] = {}  # original -> sanitized

    def populate_from_functions(self, functions: list) -> None:
        """
        Build name mappings from function definitions upfront.

        This is useful when you need to call get_original_name() before
        get_sanitized_name() (e.g., when parsing saved outputs without
        preprocessing).

        Args:
            functions: List of function definitions with "name" field
        """
        # Clear existing mappings
        self.name_mapping.clear()
        self.reverse_mapping.clear()

        # Build all mappings at once
        existing_sanitized = set()

        for func in functions:
            original_name = func.get("name")
            if original_name:
                # Sanitize and cache
                sanitized_name = self._sanitize_name(original_name, existing_sanitized)
                existing_sanitized.add(sanitized_name)

                # Register to both mappings
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
        Get sanitized version of original name (with automatic caching).

        This method checks the cache first. If the name hasn't been sanitized yet,
        it sanitizes it, registers the mapping to both caches, and returns it.

        Args:
            original_name: Original function name

        Returns:
            Sanitized function name
        """
        # Check cache first
        if original_name in self.reverse_mapping:
            return self.reverse_mapping[original_name]
        # Cache miss: sanitize and register
        # Get all currently used sanitized names to avoid collisions
        existing_sanitized = set(self.name_mapping.keys())

        # Sanitize the name
        sanitized_name = self._sanitize_name(original_name, existing_sanitized)

        # Register to both mappings (cache for future lookups)
        self.name_mapping[sanitized_name] = original_name
        self.reverse_mapping[original_name] = sanitized_name

        return sanitized_name

    def add_mapping(self, sanitized_name: str, original_name: str) -> None:
        """
        Manually add a name mapping.

        This is useful when the sanitization is done externally and you just
        want to register the mapping.

        Args:
            sanitized_name: Sanitized function name
            original_name: Original function name
        """
        self.name_mapping[sanitized_name] = original_name
        self.reverse_mapping[original_name] = sanitized_name

    def clear(self) -> None:
        """Clear all name mappings."""
        self.name_mapping.clear()
        self.reverse_mapping.clear()

    def _sanitize_name(self, name: str, existing_sanitized: set) -> str:
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
        # print("sanitized name:", sanitized)
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
