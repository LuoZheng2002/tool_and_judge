# Parsing Strategy Alignment with parse_ast.py

## Overview

All model interface implementations now follow the exact parsing strategy from `parse_ast.py`'s `raw_to_json()` function. This ensures consistent, reliable output parsing across all models.

## Parsing Strategies

### API Models (GPT-4o-mini, Claude Sonnet/Haiku, DeepSeek)

All API models follow the same parsing strategy for Python function call syntax:

```
Input Format:  [func_name1(param1=value1, param2=value2), func_name2(param3=value3)]
Output Format: [{"func_name1": {"param1": value1, "param2": value2}}, {"func_name2": {"param3": value3}}]
```

#### Parsing Steps (from parse_ast.py:170-189)

1. **Strip Backticks and Whitespace** (line 170)
   ```python
   raw_output = raw_output.strip("`\n ")
   ```

2. **Add Missing Brackets** (lines 171-174)
   ```python
   if not raw_output.startswith("["):
       raw_output = "[" + raw_output
   if not raw_output.endswith("]"):
       raw_output = raw_output + "]"
   ```

3. **Remove Wrapping Quotes** (line 176)
   ```python
   cleaned_input = raw_output.strip().strip("'")
   ```

4. **Parse as Python AST** (line 178)
   ```python
   parsed = ast.parse(cleaned_input, mode="eval")
   ```

5. **Extract Function Calls** (lines 181-189)
   - Handle single function call: `ast.Call` → resolve directly
   - Handle multiple function calls: `ast.List` of `ast.Call` nodes → resolve each

6. **Resolve AST Nodes to Python Values**
   - Uses `_resolve_ast_by_type()` for parameter values
   - Uses `_resolve_ast_call()` to construct function call dictionary
   - Handles nested attributes: `module.submodule.function_name`
   - Converts AST nodes to native Python types

**Error Handling:**
- Returns error string if syntax error: `"Failed to decode AST: Invalid syntax."`
- Returns error string if not AST.Call: `"Failed to decode AST: Expected AST Call node, but got {type}"`

**Special Cases:**
- Tuples converted to lists (to match ground truth)
- Lowercase "true"/"false" converted to Python `True`/`False`
- Unquoted identifiers preserved as strings
- Binary operations evaluated with `eval()`

---

### Granite Model (Local)

Granite model follows a different parsing strategy for JSON function call syntax:

```
Input Format:  <tool_call>[{"name": "func_name", "arguments": {"param1": value1, ...}}, ...]
Output Format: [{"func_name": {"param1": value1, ...}}, ...]
```

#### Parsing Steps (from parse_ast.py:135-166)

1. **Strip Leading/Trailing Whitespace** (line 135)
   ```python
   model_result_raw = raw_output.strip()
   ```

2. **Remove <tool_call> Wrapper** (lines 138-139)
   ```python
   if model_result_raw.startswith("<tool_call>"):
       model_result_raw = model_result_raw[len("<tool_call>"):]
   ```

3. **Strip Backticks and Whitespace** (line 141)
   ```python
   model_result_raw = model_result_raw.strip("`\n ")
   ```

4. **Add Missing Brackets** (lines 144-147)
   ```python
   if not model_result_raw.startswith("["):
       model_result_raw = "[" + model_result_raw
   if not model_result_raw.endswith("]"):
       model_result_raw = model_result_raw + "]"
   ```

5. **Parse as JSON Array** (line 151)
   ```python
   tool_calls = json.loads(model_result_raw)
   ```

6. **Convert Format** (lines 156-166)
   - Extract `"name"` and `"arguments"` from each tool call
   - Convert to format: `{func_name: func_arguments}`
   - Collect all into list

**Error Handling:**
- Returns error string if JSON decode fails: `"Failed to decode JSON: Invalid JSON format."`
- Returns error string if structure is invalid: `"Failed to decode JSON: Invalid tool call structure."`
- Returns error string if not a list: `"Failed to decode JSON: Expected a list of tool calls."`

---

## Implementation Mapping

### File Structure

| File | Model | Parse Strategy | Key Methods |
|------|-------|---|---|
| `gpt_4o_mini_interface.py` | GPT-4o-mini | API (Python AST) | `parse_output()`, `_resolve_ast_call()`, `_resolve_ast_by_type()` |
| `claude_sonnet_interface.py` | Claude Sonnet | API (Python AST) | `parse_output()`, `_resolve_ast_call()`, `_resolve_ast_by_type()` |
| `claude_haiku_interface.py` | Claude Haiku | API (Python AST) | `parse_output()`, `_resolve_ast_call()`, `_resolve_ast_by_type()` |
| `deepseek_chat_interface.py` | DeepSeek Chat | API (Python AST) | `parse_output()`, `_resolve_ast_call()`, `_resolve_ast_by_type()` |
| `granite_3_1_8b_instruct_interface.py` | Granite 3.1 8B | Local (JSON) | `parse_output()` (JSON parsing) |

### Code Reuse

**API Models (GPT-4o-mini, Claude models, DeepSeek):**
- All implement identical parsing logic
- Use same `_resolve_ast_call()` method
- Use same `_resolve_ast_by_type()` method
- Could be refactored to shared utility class (future improvement)

**Granite Model:**
- Uses JSON parsing (unique implementation)
- Simpler logic compared to API models
- Direct JSON deserialization

---

## Implementation Details

### AST Resolution (`_resolve_ast_by_type`)

Handles the following AST node types:

| AST Type | Resolution |
|----------|---|
| `ast.Constant` | Return value directly; handle `Ellipsis` → `"..."` |
| `ast.UnaryOp` | Negate operand value |
| `ast.List` | Recursively resolve each element |
| `ast.Dict` | Recursively resolve keys and values |
| `ast.NameConstant` | Return boolean value directly |
| `ast.BinOp` | Evaluate with `eval(ast.unparse(value))` |
| `ast.Name` | Handle "true"→`True`, "false"→`False`, else return ID |
| `ast.Call` | If no keywords, unpars; else recursively resolve |
| `ast.Tuple` | Convert to list (per parse_ast.py:96 comment) |
| `ast.Lambda` | Evaluate function body |
| `ast.Ellipsis` | Return `"..."` |
| `ast.Subscript` | Unparse to string representation |

### Function Call Resolution (`_resolve_ast_call`)

Handles function names with nested attributes:
```python
# Example: module.submodule.function_name
# Parses AST attributes and reconstructs as dot-separated string
```

---

## Return Format Consistency

### API Models Return Format
```python
[
    {
        "function_name": {
            "param1": value1,
            "param2": value2,
            ...
        }
    },
    {
        "another_function": {
            "param3": value3
        }
    }
]
```

### Granite Model Return Format
```python
[
    {
        "function_name": {
            "param1": value1,
            "param2": value2,
            ...
        }
    },
    {
        "another_function": {
            "param3": value3
        }
    }
]
```

**Both models produce identical output format after parsing.**

---

## Error Handling Consistency

All models follow the same error handling approach:

1. **Return Error String Instead of Exception**
   - Matches `raw_to_json()` behavior
   - String starts with `"Failed to decode ..."` pattern
   - Allows caller to distinguish parse errors from valid empty results

2. **Error Message Format**
   - API models: `"Failed to decode AST: {message}"`
   - Granite model: `"Failed to decode JSON: {message}"`

3. **Error Types**
   - Syntax errors
   - Invalid structure
   - Type mismatches
   - Missing required fields

---

## Line References to parse_ast.py

All implementations include comments referencing exact line numbers from `parse_ast.py`:

### API Models
- Line 170: Strip backticks and whitespace
- Lines 171-174: Add missing brackets
- Line 176: Remove wrapping quotes
- Line 178: Parse as Python AST
- Lines 181-189: Extract function calls
- Lines 111-119 (resolve_ast_call): Handle nested attributes
- Lines 58-108 (resolve_ast_by_type): AST node resolution

### Granite Model
- Line 135: Strip whitespace
- Lines 138-139: Remove `<tool_call>` wrapper
- Line 141: Strip backticks
- Lines 144-147: Add missing brackets
- Line 151: Parse JSON
- Lines 156-166: Convert format
- Lines 159-162: Extract name and arguments

---

## Testing & Validation

### Unit Tests Could Cover

1. **API Models:**
   - Single function call with simple parameters
   - Multiple function calls
   - Nested data structures (dicts, lists)
   - Special values (booleans, ellipsis, etc.)
   - Error cases (invalid syntax, missing brackets)

2. **Granite Model:**
   - Well-formed JSON output
   - Missing `<tool_call>` prefix
   - Extra whitespace and backticks
   - Invalid JSON structure
   - Missing required fields

### Integration Tests

1. Compare parse results with `raw_to_json()` on same inputs
2. Verify output format matches `evaluate_json()` expectations
3. Test with actual model outputs from production

---

## Future Improvements

### 1. Shared Utility Module
Extract common AST resolution logic to `models/ast_parser.py`:
```python
class ASTParser:
    @staticmethod
    def resolve_ast_call(elem: ast.Call) -> Dict[str, Dict[str, Any]]:
        ...

    @staticmethod
    def resolve_ast_by_type(value: ast.expr) -> Any:
        ...
```

Then all API models inherit from shared class:
```python
class APIModelInterface(ModelInterface, ASTParser):
    def parse_output(self, raw_output: str) -> ...:
        # Use inherited methods
```

### 2. Dedicated JSON Parser
Extract Granite JSON parsing to `models/granite_parser.py`:
```python
class GraniteJSONParser:
    @staticmethod
    def parse_granite_output(raw_output: str) -> Union[List[Dict], str]:
        ...
```

### 3. Configuration-Driven Parsing
Create a parsing configuration system:
```python
PARSING_STRATEGIES = {
    "api": {"type": "ast", "format": "python_calls"},
    "granite": {"type": "json", "format": "tool_calls"}
}
```

### 4. Enhanced Error Recovery
Add partial parsing for malformed output:
```python
def parse_output_with_fallback(self, raw_output):
    # Try primary parsing
    # If fails, attempt recovery strategies
    # Return best-effort result with confidence score
```

---

## Verification Checklist

- ✅ All API models use AST parsing strategy
- ✅ Granite model uses JSON parsing strategy
- ✅ Return formats are consistent
- ✅ Error handling follows raw_to_json() pattern
- ✅ Comments reference exact parse_ast.py line numbers
- ✅ All AST node types handled
- ✅ Special cases (tuples, booleans) handled correctly
- ✅ Nested attributes resolved correctly
- ✅ All files syntax-validated

---

## Summary

All model interfaces now implement parsing strategies that exactly mirror the proven logic in `parse_ast.py`. This ensures:

1. **Consistency**: All models produce identical output format
2. **Reliability**: Uses battle-tested parsing logic
3. **Maintainability**: Easy to understand and update
4. **Debuggability**: Line references for quick correlation
5. **Extensibility**: New models can follow the same pattern
