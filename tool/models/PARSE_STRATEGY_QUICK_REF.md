# Parsing Strategy Quick Reference

## Side-by-Side Comparison

### API Models (GPT-4o, Claude, DeepSeek)

**Input:**
```python
"[func_name(param1=5, param2='hello')]"
```

**Processing Steps:**
```
1. Strip backticks/whitespace:  "[func_name(param1=5, param2='hello')]"
2. Add brackets if needed:      "[func_name(param1=5, param2='hello')]"  (no change)
3. Remove quotes:               "[func_name(param1=5, param2='hello')]"  (no change)
4. Parse as Python AST
5. Extract from ast.Call nodes
6. Resolve AST values to Python
```

**Output:**
```python
[{"func_name": {"param1": 5, "param2": "hello"}}]
```

**from parse_ast.py:**
- Lines 170-176: Cleanup
- Line 178: AST parsing
- Lines 181-189: Extraction & resolution

---

### Granite Model (Local)

**Input:**
```python
"<tool_call>[{\"name\": \"func_name\", \"arguments\": {\"param1\": 5, \"param2\": \"hello\"}}]"
```

**Processing Steps:**
```
1. Strip whitespace:            "<tool_call>[{...}]"
2. Remove <tool_call> prefix:   "[{...}]"
3. Strip backticks:             "[{...}]"
4. Add brackets if needed:      "[{...}]"  (no change)
5. Parse as JSON
6. Extract name & arguments
7. Reformat to {func_name: args}
```

**Output:**
```python
[{"func_name": {"param1": 5, "param2": "hello"}}]
```

**from parse_ast.py:**
- Lines 135-141: Cleanup & remove wrapper
- Lines 144-147: Add brackets
- Line 151: JSON parsing
- Lines 156-166: Format conversion

---

## Error Handling Pattern

Both strategies return error strings instead of raising exceptions:

```python
def parse_output(self, raw_output: str) -> Union[List[Dict], str]:
    try:
        # Parsing logic
        parsed = ...
    except SyntaxError:
        return "Failed to decode AST: Invalid syntax."
    except ValueError:
        return "Failed to decode JSON: Invalid format."

    # Validation
    if not isinstance(parsed, list):
        return f"Failed to decode {format}: Expected list but got {type}"

    return parsed
```

---

## Output Format (Identical for All Models)

```python
[
    {
        "function_name_1": {
            "param_name_1": value_1,
            "param_name_2": value_2,
        }
    },
    {
        "function_name_2": {
            "param_name_1": value_1,
        }
    }
]
```

**Key Points:**
- List of dictionaries
- Each dict has exactly one key (function name)
- Value is dict of arguments
- Arguments are Python-typed values

---

## File Mapping

| File | parse_ast.py Section | Strategy |
|------|---|---|
| `gpt_4o_mini_interface.py` | Lines 169-191 (else branch) | Python AST |
| `claude_sonnet_interface.py` | Lines 169-191 (else branch) | Python AST |
| `claude_haiku_interface.py` | Lines 169-191 (else branch) | Python AST |
| `deepseek_chat_interface.py` | Lines 169-191 (else branch) | Python AST |
| `granite_3_1_8b_instruct_interface.py` | Lines 131-167 (if branch) | JSON |

---

## Method Reference

### For API Models

```python
# In your model interface class:

def parse_output(self, raw_output: str) -> Union[List[Dict], str]:
    # Implementation follows lines 170-189 of parse_ast.py
    # Uses: _resolve_ast_call() and _resolve_ast_by_type()

def _resolve_ast_call(self, elem: ast.Call) -> Dict[str, Dict]:
    # Implementation of parse_ast.py's resolve_ast_call (lines 110-124)
    # Returns: {"func_name": {"arg1": val1, ...}}

def _resolve_ast_by_type(self, value: ast.expr) -> Any:
    # Implementation of parse_ast.py's resolve_ast_by_type (lines 58-108)
    # Returns: Native Python value
```

### For Granite Model

```python
# In your model interface class:

def parse_output(self, raw_output: str) -> Union[List[Dict], str]:
    # Implementation follows lines 135-166 of parse_ast.py
    # Direct JSON parsing, no helper methods needed
```

---

## Common Issues & Fixes

### Issue 1: Model adds extra backticks
```python
# Input:
"`[func(x=5)]`"

# Fix already handles this:
raw_output.strip("`\n ")  # Removes all surrounding backticks
```

### Issue 2: Missing brackets
```python
# Input:
"func(x=5)"

# Fix already handles this:
if not raw_output.startswith("["):
    raw_output = "[" + raw_output
if not raw_output.endswith("]"):
    raw_output = raw_output + "]"
# Result: "[func(x=5)]"
```

### Issue 3: Wrapping quotes
```python
# Input:
"'[func(x=5)]'"

# Fix already handles this:
cleaned_input = raw_output.strip().strip("'")
# Result: "[func(x=5)]"
```

### Issue 4: Invalid AST syntax
```python
# Input contains unclosed paren:
"[func(x=5]"

# Handler:
try:
    parsed = ast.parse(cleaned_input, mode="eval")
except SyntaxError:
    return "Failed to decode AST: Invalid syntax."
```

### Issue 5: Wrong tool call structure (Granite)
```python
# Input:
'[{"missing_name_field": {...}}]'

# Handler:
if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
    # OK
else:
    return "Failed to decode JSON: Invalid tool call structure."
```

---

## Testing Template

```python
def test_parse_output():
    interface = GPT4oMiniInterface()  # or any API model

    # Test 1: Single function call
    result = interface.parse_output("[func(x=5, y='hello')]")
    assert result == [{"func": {"x": 5, "y": "hello"}}]

    # Test 2: Multiple function calls
    result = interface.parse_output("[func1(x=5), func2(y=10)]")
    assert result == [{"func1": {"x": 5}}, {"func2": {"y": 10}}]

    # Test 3: With backticks
    result = interface.parse_output("`[func(x=5)]`")
    assert result == [{"func": {"x": 5}}]

    # Test 4: Without brackets
    result = interface.parse_output("func(x=5)")
    assert result == [{"func": {"x": 5}}]

    # Test 5: Error case
    result = interface.parse_output("[func(")
    assert isinstance(result, str)
    assert "Failed to decode" in result


def test_granite_parse_output():
    interface = Granite3_1_8BInstructInterface()

    # Test 1: Standard Granite output
    result = interface.parse_output(
        '<tool_call>[{"name": "func", "arguments": {"x": 5}}]'
    )
    assert result == [{"func": {"x": 5}}]

    # Test 2: With extra whitespace
    result = interface.parse_output(
        '  <tool_call>  [{"name": "func", "arguments": {"x": 5}}]  '
    )
    assert result == [{"func": {"x": 5}}]

    # Test 3: Error case
    result = interface.parse_output('[{"invalid": "structure"}]')
    assert isinstance(result, str)
    assert "Invalid tool call structure" in result
```

---

## Integration with main.py

```python
from models.model_factory import create_model_interface

# Create interface
model_interface = create_model_interface(model)

# Get raw output
raw_output = model_interface.infer(system_prompt, user_query)

# Parse output
parsed = model_interface.parse_output(raw_output)

# Handle results
if isinstance(parsed, str):
    # It's an error message
    print(f"Parse error: {parsed}")
else:
    # It's valid parsed output
    for func_call in parsed:
        func_name = next(iter(func_call))
        arguments = func_call[func_name]
        print(f"Function: {func_name}, Args: {arguments}")
```

---

## Key Differences Summarized

| Aspect | API Models | Granite |
|---|---|---|
| **Input Format** | Python function calls | JSON with metadata |
| **Parser** | ast.parse() | json.loads() |
| **Wrapper Tag** | None | `<tool_call>` |
| **Structure** | `[func(x=y)]` | `[{"name": "...", "arguments": {...}}]` |
| **Complexity** | Higher (AST resolution) | Lower (direct JSON) |
| **parse_ast.py Section** | Lines 169-191 | Lines 131-167 |

---

**All implementations verified against parse_ast.py source code with exact line references.**
