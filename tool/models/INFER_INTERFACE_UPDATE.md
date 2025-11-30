# Infer Interface Update: Functions + User Query

## Summary

All model interfaces have been updated to accept `functions` and `user_query` directly, rather than requiring a pre-generated `system_prompt`. The system prompt is now generated internally using logic from `main.py`'s `gen_developer_prompt()` function.

## New Interface Signature

### All Models

```python
def infer(self, functions: List[Dict[str, Any]], user_query: str,
          prompt_passing_in_english: bool = True, model=None) -> str:
    """
    Args:
        functions: List of available function definitions in JSON format
        user_query: User query/question as a string
        prompt_passing_in_english: Whether to request English parameter passing (default: True)
        model: Optional model type (kept for interface compatibility, used by Granite)

    Returns:
        Raw model output as a string
    """
```

## Changes by Model Type

### API Models (GPT-4o-mini, Claude Sonnet/Haiku, DeepSeek)

**Before:**
```python
interface = create_model_interface(ApiModel.GPT_4O_MINI)
result = interface.infer(
    system_prompt="custom prompt",
    user_query="What's 2+2?"
)
```

**After:**
```python
interface = create_model_interface(ApiModel.GPT_4O_MINI)
result = interface.infer(
    functions=[{"name": "add", ...}],
    user_query="What's 2+2?"
)
```

**Internal Changes:**
- Added `_generate_system_prompt()` method
- Automatically generates API-specific prompt format
- Supports `prompt_passing_in_english` parameter
- Produces Python function call syntax prompt

### Granite Model (Local)

**Before:**
```python
interface = create_model_interface(config.model, generator=gen)
result = interface.infer_with_functions(
    system_prompt="custom prompt",
    user_query="What's 2+2?",
    functions=[...]
)
```

**After:**
```python
interface = create_model_interface(config.model, generator=gen)
result = interface.infer(
    functions=[{"name": "add", ...}],
    user_query="What's 2+2?"
)
```

**Internal Changes:**
- Updated `infer()` to accept functions directly
- Added `_generate_system_prompt()` method
- Automatically generates Granite-specific prompt format
- Updated batch methods: `infer_batch()`, `infer_batch_with_functions()`
- Maintains backward compatibility with explicit system prompt methods

## System Prompt Generation

Both API and Granite models now use the same logic as `main.py`'s `gen_developer_prompt()`:

### API Models
```python
def _generate_system_prompt(self, functions, prompt_passing_in_english=True, is_granite=False):
    # JSON dumps functions
    # Creates "API format" prompt requesting Python function call syntax
    # Returns: System prompt string
```

**Prompt Template:**
```
You are an expert in composing functions. [... instructions ...]

You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of
[func_name1(params_name1=params_value1...), func_name2(...)].
You SHOULD NOT include any other text in the response. [pass_in_english]

Here is a list of functions in json format that you can invoke.
[function JSON]
```

### Granite Model
```python
def _generate_system_prompt(self, functions, prompt_passing_in_english=True):
    # JSON dumps functions
    # Creates "Granite format" prompt requesting JSON function call syntax
    # Returns: System prompt string
```

**Prompt Template:**
```
You are an expert in composing functions. [... instructions ...]

You should only return the function calls in your response, in JSON format as a list
where each element has the format {"name": "function_name", "arguments": {...}}.
[pass_in_english]

Here is a list of functions in json format that you can invoke.
[function JSON]
```

## Interface Methods by Model

### API Models
- `infer()` - ✅ Updated to new signature
- `parse_output()` - Unchanged
- Helper methods: `_generate_system_prompt()`, `_resolve_ast_call()`, `_resolve_ast_by_type()`

### Granite Model
- `infer()` - ✅ Updated to new signature
- `infer_batch()` - ✅ Updated to accept functions lists
- `infer_with_functions()` - Unchanged (accepts explicit system prompt)
- `infer_batch_with_functions()` - Unchanged (accepts explicit system prompts)
- `parse_output()` - Unchanged
- Helper methods: `_generate_system_prompt()`, `_format_granite_chat_template()`

## Usage Examples

### API Model
```python
from models.model_factory import create_model_interface
from config import ApiModel
import json

# Load functions
with open("function_call.txt") as f:
    functions = json.load(f)

# Create interface
interface = create_model_interface(ApiModel.GPT_4O_MINI)

# Run inference
raw_output = interface.infer(
    functions=functions,
    user_query="Can I find the dimensions of a triangle with sides 3, 4, 5?",
    prompt_passing_in_english=True
)

# Parse output
parsed = interface.parse_output(raw_output)
print(parsed)
# Output: [{"triangle_properties.get": {"side1": 3, "side2": 4, "side3": 5}}]
```

### Granite Model
```python
from models.model_factory import create_model_interface
from config import LocalModelStruct, LocalModel
from call_llm import make_chat_pipeline
import json

# Load functions
with open("function_call.txt") as f:
    functions = json.load(f)

# Create generator and interface
generator = make_chat_pipeline(LocalModel.GRANITE_3_1_8B_INSTRUCT)
config = LocalModelStruct(model=LocalModel.GRANITE_3_1_8B_INSTRUCT, generator=generator)
interface = create_model_interface(config, generator=generator)

# Run inference
raw_output = interface.infer(
    functions=functions,
    user_query="Can I find the dimensions of a triangle with sides 3, 4, 5?",
    prompt_passing_in_english=True
)

# Parse output
parsed = interface.parse_output(raw_output)
print(parsed)
# Output: [{"triangle_properties.get": {"side1": 3, "side2": 4, "side3": 5}}]
```

### Batch Processing (Granite)
```python
# Batch inference with functions
functions_list = [functions, functions, functions]
user_queries = [
    "Calculate triangle properties for sides 3, 4, 5",
    "Calculate circle area for radius 5",
    "Get circle circumference for radius 10"
]

batch_results = interface.infer_batch(
    functions_list=functions_list,
    user_queries=user_queries,
    prompt_passing_in_english=True
)

for result in batch_results:
    parsed = interface.parse_output(result)
    print(parsed)
```

## Changes to main.py (Recommended)

The refactored interfaces simplify main.py's inference logic:

### Old approach:
```python
model_interface = create_model_interface(model)
result = model_interface.infer(system_prompt=developer_prompt, user_query=user_question)
```

### New approach:
```python
model_interface = create_model_interface(model)
result = model_interface.infer(
    functions=functions,
    user_query=user_question,
    prompt_passing_in_english=True
)
```

This eliminates the need for `gen_developer_prompt()` in main.py when using the interfaces directly.

## Function Format Reference

Functions are passed as a list of dictionaries (from function_call.txt):

```python
[
    {
        "name": "triangle_properties.get",
        "description": "Retrieve triangle dimensions...",
        "parameters": {
            "type": "dict",
            "properties": {
                "side1": {"type": "integer", "description": "..."},
                "side2": {"type": "integer", "description": "..."},
                "side3": {"type": "integer", "description": "..."},
                "get_area": {"type": "boolean", ...},
                ...
            },
            "required": ["side1", "side2", "side3"]
        }
    },
    {
        "name": "circle_properties.get",
        ...
    }
]
```

## API Compatibility

All interfaces maintain backward compatibility:

- `infer()` - **New signature** (breaking change - now requires functions/user_query)
- `parse_output()` - Unchanged
- `infer_with_functions()` - Still available on Granite
- `infer_batch_with_functions()` - Still available on Granite
- All helper methods preserved

## File-by-File Updates

| File | Changes |
|------|---------|
| models/base.py | Updated abstract `infer()` signature |
| models/gpt_4o_mini_interface.py | `infer()` signature, added `_generate_system_prompt()` |
| models/claude_sonnet_interface.py | `infer()` signature, added `_generate_system_prompt()` |
| models/claude_haiku_interface.py | `infer()` signature, added `_generate_system_prompt()` |
| models/deepseek_chat_interface.py | `infer()` signature, added `_generate_system_prompt()` |
| models/granite_3_1_8b_instruct_interface.py | `infer()` signature updated, `infer_batch()` updated, added `_generate_system_prompt()` |
| models/model_factory.py | Unchanged |

## Testing Checklist

- [ ] Test API models with new `infer()` signature
- [ ] Test Granite with new `infer()` signature
- [ ] Test Granite `infer_batch()` with functions lists
- [ ] Test backward compatibility with `infer_with_functions()`
- [ ] Verify system prompt generation for API models
- [ ] Verify system prompt generation for Granite
- [ ] Test with actual function definitions from function_call.txt
- [ ] Verify parsing works on generated outputs
- [ ] Test `prompt_passing_in_english` parameter

## Migration Guide for main.py

To fully utilize the new interfaces in main.py:

1. **Update inference() function:**
   ```python
   def inference(model: Model, test_entry: dict):
       functions = test_entry['function']
       user_question = test_entry["question"][0][0]['content']

       # Create interface
       model_interface = create_model_interface(model)

       # Inference now accepts functions directly
       result = model_interface.infer(
           functions=functions,
           user_query=user_question,
           prompt_passing_in_english=True
       )

       return {"id": test_entry["id"], "result": result}
   ```

2. **Remove gen_developer_prompt() call** - It's now internal to infer()

3. **Simplify batch processing:**
   ```python
   # API models: sequential calls
   model_interface = create_model_interface(config.model)
   batch_results = [
       model_interface.infer(functions, user_query, prompt_passing_in_english=True)
       for functions, user_query in batch_data
   ]

   # Granite: true batch processing
   model_interface = create_model_interface(config.model, generator=config.model.generator)
   batch_results = model_interface.infer_batch(
       functions_list=functions_list,
       user_queries=user_queries,
       prompt_passing_in_english=True
   )
   ```

## Summary

This update makes the model interfaces cleaner and more intuitive by:
1. Accepting the domain data (functions) directly
2. Generating appropriate system prompts internally
3. Reducing boilerplate in main.py
4. Maintaining all existing functionality
5. Supporting the full range of generation strategies from parse_ast.py

All implementations follow the exact prompt generation logic from `main.py`'s `gen_developer_prompt()` function.
