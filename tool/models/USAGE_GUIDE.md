# Model Interface Usage Guide

## Quick Start

### 1. Import the Factory
```python
from models.model_factory import create_model_interface
from config import ApiModel, LocalModel, LocalModelStruct
```

### 2. Create an Interface

**For API Models:**
```python
# GPT-4o-mini
model = create_model_interface(ApiModel.GPT_4O_MINI)

# Claude Sonnet
model = create_model_interface(ApiModel.CLAUDE_SONNET)

# Claude Haiku
model = create_model_interface(ApiModel.CLAUDE_HAIKU)

# DeepSeek Chat
model = create_model_interface(ApiModel.DEEPSEEK_CHAT)
```

**For Local Models:**
```python
# Granite (requires generator from make_chat_pipeline)
from call_llm import make_chat_pipeline

generator = make_chat_pipeline(LocalModel.GRANITE_3_1_8B_INSTRUCT)
config = LocalModelStruct(model=LocalModel.GRANITE_3_1_8B_INSTRUCT, generator=generator)
model_interface = create_model_interface(config, generator=generator)
```

### 3. Run Inference

**Simple Inference:**
```python
result = model_interface.infer(
    system_prompt="You are a helpful assistant.",
    user_query="What is the capital of France?"
)
# Returns: str (raw model output)
```

**With Function Definitions (Granite only):**
```python
functions = [
    {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "dict",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
]

result = model_interface.infer_with_functions(
    system_prompt="You are a weather assistant.",
    user_query="What's the weather in London?",
    functions=functions
)
# Returns: str (raw model output with function calls)
```

### 4. Parse Output

```python
parsed_calls = model_interface.parse_output(result)
# Returns: List[Dict[str, Any]] in format:
# [
#     {
#         "name": "get_weather",
#         "arguments": {"location": "London"}
#     },
#     ...
# ]
```

## Complete Example

### Single Inference Pipeline
```python
from models.model_factory import create_model_interface
from config import ApiModel

# Create interface
model = create_model_interface(ApiModel.GPT_4O_MINI)

# System and user prompts
system_prompt = """You are an expert in composing functions.
Return function calls in Python syntax: [func_name(param1=value1)]"""

user_query = "Call get_user with id=123"

# Run inference
raw_output = model.infer(system_prompt=system_prompt, user_query=user_query)
print("Raw output:", raw_output)

# Parse output
parsed = model.parse_output(raw_output)
print("Parsed calls:", parsed)
# Output: [{"name": "get_user", "arguments": {"id": 123}}]
```

### Batch Processing with Granite
```python
from models.model_factory import create_model_interface
from config import LocalModel, LocalModelStruct
from call_llm import make_chat_pipeline

# Setup
generator = make_chat_pipeline(LocalModel.GRANITE_3_1_8B_INSTRUCT)
config = LocalModelStruct(model=LocalModel.GRANITE_3_1_8B_INSTRUCT, generator=generator)
model = create_model_interface(config, generator=generator)

functions = [{"name": "calculate", "description": "...", ...}]

# Batch processing
system_prompts = [
    "You are a calculator assistant.",
    "You are a calculator assistant."
]
user_queries = [
    "Calculate 5 + 3",
    "Calculate 10 * 2"
]
batch_functions = [functions, functions]

# Run batch
results = model.infer_batch_with_functions(
    system_prompts=system_prompts,
    user_queries=user_queries,
    batch_functions=batch_functions
)

# Parse results
for i, raw_result in enumerate(results):
    parsed = model.parse_output(raw_result)
    print(f"Result {i}: {parsed}")
```

## Method Reference

### All Models

#### `infer(system_prompt: str, user_query: str) -> str`
Run single inference.

**Parameters:**
- `system_prompt`: System message as string
- `user_query`: User message as string

**Returns:** Raw model output as string

**Example:**
```python
result = model.infer("Be helpful", "Hello!")
```

---

#### `parse_output(raw_output: str) -> List[Dict[str, Any]]`
Parse model output to function calls.

**Parameters:**
- `raw_output`: Raw string from `infer()`

**Returns:** List of function call dicts:
```python
[
    {
        "name": "function_name",
        "arguments": {"param1": value1, "param2": value2}
    },
    ...
]
```

**Example:**
```python
calls = model.parse_output(result)
for call in calls:
    print(f"Function: {call['name']}")
    print(f"Arguments: {call['arguments']}")
```

---

### Granite Model Only

#### `infer_with_functions(system_prompt: str, user_query: str, functions: List[Dict]) -> str`
Run inference with function definitions.

**Parameters:**
- `system_prompt`: System message
- `user_query`: User message
- `functions`: List of function definitions (JSON format)

**Returns:** Raw model output with function calls

**Example:**
```python
functions = [
    {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
            "type": "dict",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    }
]

result = granite_model.infer_with_functions(
    system_prompt="You are a math assistant",
    user_query="Add 5 and 3",
    functions=functions
)
```

---

#### `infer_batch(system_prompts: List[str], user_queries: List[str]) -> List[str]`
Run batch inference without functions.

**Parameters:**
- `system_prompts`: List of system prompts
- `user_queries`: List of user queries

**Returns:** List of raw outputs

**Example:**
```python
results = granite_model.infer_batch(
    system_prompts=["Assistant"] * 2,
    user_queries=["Query 1", "Query 2"]
)
```

---

#### `infer_batch_with_functions(system_prompts: List[str], user_queries: List[str], batch_functions: List[List[Dict]]) -> List[str]`
Run batch inference with functions.

**Parameters:**
- `system_prompts`: List of system prompts
- `user_queries`: List of user queries
- `batch_functions`: List of function lists (one per query)

**Returns:** List of raw outputs with function calls

**Example:**
```python
functions = [...]  # Common function definitions

results = granite_model.infer_batch_with_functions(
    system_prompts=["Assistant"] * 3,
    user_queries=["Q1", "Q2", "Q3"],
    batch_functions=[functions] * 3
)
```

---

## Output Format Details

### API Models Output
Raw output example:
```
[func_name(param1=value1, param2=value2), another_func(param3=value3)]
```

Parsed format:
```python
[
    {"name": "func_name", "arguments": {"param1": value1, "param2": value2}},
    {"name": "another_func", "arguments": {"param3": value3}}
]
```

### Granite Model Output
Raw output example:
```
<tool_call>[{"name": "func_name", "arguments": {"param1": value1}}, ...]
```

Parsed format:
```python
[
    {"name": "func_name", "arguments": {"param1": value1}},
    ...
]
```

## Error Handling

### Common Errors

**1. API Key Missing**
```python
EnvironmentError: OPENAI_API_KEY not found in .env
```
Solution: Ensure `.env` file has `OPENAI_API_KEY=...`

**2. Generator Not Initialized (Granite)**
```python
RuntimeError: Generator not initialized. Call set_generator() first.
```
Solution: Provide generator when creating interface:
```python
model = create_model_interface(config, generator=my_generator)
```

**3. Parse Error**
```python
ValueError: Could not find function calls in output: ...
```
Solution: Model output doesn't match expected format. Check:
- Model system prompt is correct
- Model is responding with function calls
- Output parsing matches model type

### Handling Errors
```python
from models.model_factory import create_model_interface

try:
    model = create_model_interface(ApiModel.GPT_4O_MINI)
    result = model.infer(system_prompt, user_query)
    parsed = model.parse_output(result)
except EnvironmentError as e:
    print(f"Configuration error: {e}")
except ValueError as e:
    print(f"Parsing error: {e}")
except Exception as e:
    print(f"Inference error: {e}")
```

## Tips and Best Practices

### 1. Reuse Interfaces
```python
# Good: Create once, reuse
model = create_model_interface(ApiModel.GPT_4O_MINI)
for query in queries:
    result = model.infer(system_prompt, query)

# Avoid: Creating new interface for each call
for query in queries:
    model = create_model_interface(ApiModel.GPT_4O_MINI)  # Inefficient
    result = model.infer(system_prompt, query)
```

### 2. Batch Processing for Granite
```python
# Good: Use batch methods for efficiency
results = granite_model.infer_batch_with_functions(
    system_prompts=prompts,
    user_queries=queries,
    batch_functions=functions_list
)

# Avoid: Individual calls in loop
results = []
for prompt, query, funcs in zip(prompts, queries, functions_list):
    result = granite_model.infer_with_functions(prompt, query, funcs)
    results.append(result)
```

### 3. Error Recovery
```python
# Handle partial failures gracefully
for i, raw_output in enumerate(results):
    try:
        parsed = model.parse_output(raw_output)
        process(parsed)
    except ValueError:
        print(f"Failed to parse result {i}")
        log_error(raw_output)
        continue
```

### 4. Model Selection
```python
# Match output format to model:
# - API models → Python function call syntax
# - Granite → JSON function format in system prompt

# API model system prompt
api_prompt = """Return function calls in format: [func(param=value)]"""

# Granite system prompt
granite_prompt = """Return JSON list: [{"name": "func", "arguments": {...}}]"""
```

## Troubleshooting

### Interface Creation Fails
```python
# Check model type
from config import ApiModel, LocalModel, LocalModelStruct

# Is it an ApiModel enum?
print(isinstance(config.model, ApiModel))

# Is it a LocalModelStruct?
print(isinstance(config.model, LocalModelStruct))
```

### Inference Returns Empty String
```python
# Check API key
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"API key present: {bool(api_key)}")

# Check model availability
# - API models need internet connection
# - Local models need GPU memory
```

### Parsing Fails
```python
# Print raw output to debug
result = model.infer(system_prompt, user_query)
print("Raw output:")
print(repr(result))  # Show whitespace/special chars

# Check format matches expectation
if "[" in result and "]" in result:
    # Looks like function call syntax
    pass
else:
    # Model not returning function calls
    print("Model output doesn't contain function calls")
```

## Next Steps

1. Review the `base.py` file to understand the interface contract
2. Look at individual `{model_name}_interface.py` files for implementation details
3. Check `model_factory.py` to add support for new models
4. See `REFACTORING_SUMMARY.md` for architectural details
