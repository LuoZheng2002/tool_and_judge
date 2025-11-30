"""
Model-specific interface modules for the BFCL inference pipeline.

Each model has a dedicated interface file that implements:
1. infer(system_prompt: str, user_query: str) -> str
2. parse_output(raw_output: str) -> list[dict]
"""
