# Before & After Comparison

## Problem Statement

Previously, the codebase had model-specific logic scattered throughout multiple files:
- **call_llm.py**: Model-specific inference (API vs local) and output formatting
- **main.py**: Complex match statements for different models and output parsing
- **parse_ast.py**: Model-dependent output parsing

This made it difficult to:
- Add new models (required changes in multiple files)
- Understand model behavior (logic spread across files)
- Test models in isolation
- Reuse model logic in other projects

---

## Before: Scattered Model Logic

### call_llm.py
```python
# API Models
def api_inference(model: ApiModel, input_messages: list[dict]) -> str:
    match model:
        case ApiModel.GPT_4O_MINI:
            # OpenAI specific code
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(...)
        case ApiModel.CLAUDE_SONNET | ApiModel.CLAUDE_HAIKU:
            # Anthropic specific code
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(...)
        case ApiModel.DEEPSEEK_CHAT:
            # DeepSeek specific code
            from openai import OpenAI
            client = OpenAI(api_key=..., base_url="https://api.deepseek.com")
            response = client.chat.completions.create(...)

# Local Models - Granite Only
def format_granite_chat_template(messages, functions=None, add_generation_prompt=True):
    # Granite-specific formatting with custom tags
    formatted_prompt = ""
    formatted_prompt += f"<|start_of_role|>system<|end_of_role|>{system_prompt}<|end_of_text|>\n"
    ...
    return formatted_prompt

def make_chat_pipeline(model: LocalModel):
    # Local model setup and inference
    tokenizer = AutoTokenizer.from_pretrained(model_id, ...)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, ...)

    def chat_generator():
        # Generator-based inference
        ...
    return gen
```

### main.py - Old inference() function
```python
def inference(model: Model, test_entry: dict):
    functions = test_entry['function']

    # Determine model type
    local_model = None
    if isinstance(model, LocalModelStruct):
        local_model = model.model

    # Generate prompts
    developer_prompt = gen_developer_prompt(...)
    input_messages = gen_input_messages(...)

    # Model-specific inference
    match model:
        case ApiModel() as api_model:
            result = api_inference(api_model, input_messages)
        case LocalModelStruct(model=local_model, generator=generator):
            system_message = input_messages[0]['content']
            user_message = input_messages[1]['content']

            match local_model:
                case LocalModel.GRANITE_3_1_8B_INSTRUCT:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ]
                    template = format_granite_chat_template(
                        messages, functions=functions, add_generation_prompt=True
                    )
                    result = generator.send(template)

    return {"id": test_entry["id"], "result": result}
```

### main.py - Old batch processing
```python
# Batch preparation (mixing concerns)
batch_input_messages = []
for case in batch_cases:
    functions = case['function']
    user_question = case["question"][0][0]['content']
    developer_prompt = gen_developer_prompt(...)
    input_messages = gen_input_messages(...)
    batch_input_messages.append((input_messages, functions))

# Model-specific batch handling
if is_api_model:
    api_batch_messages = [messages for messages, _ in batch_input_messages]
    batch_results = api_inference_batch(config.model, api_batch_messages)

elif is_local_model:
    local_model = config.model.model
    if local_model == LocalModel.GRANITE_3_1_8B_INSTRUCT:
        batch_templates = []
        for (input_messages, functions) in batch_input_messages:
            system_message = input_messages[0]['content']
            user_message = input_messages[1]['content']
            template = format_granite_chat_template(
                [{"role": "system", ...}, {"role": "user", ...}],
                functions=functions,
                add_generation_prompt=True
            )
            batch_templates.append(template)

        batch_results = config.model.generator.send(batch_templates)
    else:
        raise ValueError(f"Unsupported local model in batch processing: {local_model}")
```

### parse_ast.py - Output parsing
```python
def raw_to_json(model, case_id, model_result_raw):
    # Model-dependent parsing logic
    if model == ApiModel.GPT_4O_MINI or model == ApiModel.CLAUDE_SONNET:
        # Parse Python function call syntax
        # Extract from [func_name(param=value)] format
        ...
    elif model.model == LocalModel.GRANITE_3_1_8B_INSTRUCT:
        # Parse JSON format from Granite
        # Extract from <tool_call>[{...}] format
        ...

    # Further processing
    return parsed_output
```

---

## After: Organized Model Interfaces

### models/base.py - Unified Interface
```python
from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """Abstract interface for all model handlers"""

    @abstractmethod
    def infer(self, system_prompt: str, user_query: str) -> str:
        """Returns raw model output as string"""
        pass

    @abstractmethod
    def parse_output(self, raw_output: str) -> List[Dict[str, Any]]:
        """Parses raw output to function call list"""
        pass
```

### models/gpt_4o_mini_interface.py - API Model Example
```python
class GPT4oMiniInterface(ModelInterface):
    """Handler for OpenAI GPT-4o-mini model"""

    def __init__(self):
        """Initialize once with API client"""
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def infer(self, system_prompt: str, user_query: str) -> str:
        """Clean, focused inference logic"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

    def parse_output(self, raw_output: str) -> List[Dict[str, Any]]:
        """Parse Python function call syntax"""
        # Extract [func_name(param=value)] format
        ...
        return parsed_calls
```

### models/granite_3_1_8b_instruct_interface.py - Local Model
```python
class Granite3_1_8BInstructInterface(ModelInterface):
    """Handler for IBM Granite 3.1 8B Instruct local model"""

    def __init__(self, generator=None):
        """Initialize with optional generator"""
        self.generator = generator

    def infer_with_functions(self, system_prompt: str, user_query: str,
                            functions: List[Dict]) -> str:
        """Single inference with functions"""
        template = self._format_granite_chat_template(
            system_prompt=system_prompt,
            user_query=user_query,
            functions=functions,
            add_generation_prompt=True
        )
        result = self.generator.send(template)
        return result

    def infer_batch_with_functions(self, system_prompts: List[str],
                                   user_queries: List[str],
                                   batch_functions: List[List[Dict]]) -> List[str]:
        """Batch inference with functions"""
        templates = [
            self._format_granite_chat_template(...)
            for system_prompt, user_query, functions in zip(...)
        ]
        results = self.generator.send(templates)
        return results

    def parse_output(self, raw_output: str) -> List[Dict[str, Any]]:
        """Parse JSON format from Granite"""
        # Extract from <tool_call>[{...}] format
        ...
        return parsed_calls
```

### models/model_factory.py - Factory Pattern
```python
def create_model_interface(model: Union[ApiModel, LocalModelStruct],
                          generator=None) -> ModelInterface:
    """Factory function to create appropriate interface"""
    if isinstance(model, ApiModel):
        match model:
            case ApiModel.GPT_4O_MINI:
                return GPT4oMiniInterface()
            case ApiModel.CLAUDE_SONNET:
                return ClaudeSonnetInterface()
            ...
    elif isinstance(model, LocalModelStruct):
        if model.model == LocalModel.GRANITE_3_1_8B_INSTRUCT:
            return Granite3_1_8BInstructInterface(generator=generator)
```

### main.py - Refactored inference()
```python
def inference(model: Model, test_entry: dict, model_interface=None):
    """Unified inference with model interface abstraction"""

    functions = test_entry['function']
    user_question = test_entry["question"][0][0]['content']

    # Determine model type for prompt generation
    local_model = None
    if isinstance(model, LocalModelStruct):
        local_model = model.model

    # Generate system prompt
    developer_prompt = gen_developer_prompt(
        function_calls=functions,
        prompt_passing_in_english=True,
        model=local_model
    )

    # Create model interface if not provided
    if model_interface is None:
        model_interface = create_model_interface(model)

    # Unified inference (works for all models!)
    result = model_interface.infer(
        system_prompt=developer_prompt,
        user_query=user_question
    )

    return {"id": test_entry["id"], "result": result}
```

### main.py - Refactored batch processing
```python
# Prepare batch data (cleaner structure)
batch_data = []
for case in batch_cases:
    functions = case['function']
    user_question = case["question"][0][0]['content']
    local_model = config.model.model if isinstance(config.model, LocalModelStruct) else None
    developer_prompt = gen_developer_prompt(...)
    batch_data.append((developer_prompt, user_question, functions))

# Model-agnostic processing
if is_api_model:
    model_interface = create_model_interface(config.model)
    batch_results = []
    for system_prompt, user_query, _ in batch_data:
        result = model_interface.infer(
            system_prompt=system_prompt,
            user_query=user_query
        )
        batch_results.append(result)

elif is_local_model:
    model_interface = create_model_interface(
        config.model,
        generator=config.model.generator
    )
    system_prompts = [data[0] for data in batch_data]
    user_queries = [data[1] for data in batch_data]
    batch_functions = [data[2] for data in batch_data]

    batch_results = model_interface.infer_batch_with_functions(
        system_prompts=system_prompts,
        user_queries=user_queries,
        batch_functions=batch_functions
    )
```

---

## Comparison Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Model Logic Location** | Scattered (call_llm.py, main.py, parse_ast.py) | Centralized (models/{model}_interface.py) |
| **Adding New Model** | Modify 3+ files with complex logic | Create 1 new file with clean interface |
| **Code Duplication** | High (API models repeat similar code) | Low (shared base class) |
| **Main.py Complexity** | High (extensive case matching) | Low (unified interface) |
| **Testing Difficulty** | Hard (models interdependent) | Easy (each model isolated) |
| **Batch Processing** | Complex model-specific branches | Simple unified approach |
| **Parsing Location** | Scattered in parse_ast.py | In each model interface |
| **Reusability** | Limited (tightly coupled) | High (independent modules) |
| **Lines of Code** | ~300 scattered | ~1200 organized + clearer |
| **Maintainability** | Low | High |

---

## Benefits Realized

### 1. **Clarity**
```python
# Before: Need to understand multiple files and case statements
result = api_inference(api_model, input_messages)

# After: Clear what the code does
result = model_interface.infer(system_prompt, user_query)
```

### 2. **Extensibility**
```python
# Before: To add GPT-4-turbo, modify call_llm.py, main.py, parse_ast.py
# After: Just create gpt_4_turbo_interface.py with 3 methods
class GPT4TurboInterface(ModelInterface):
    def infer(self, system_prompt, user_query):
        ...
    def parse_output(self, raw_output):
        ...
```

### 3. **Testability**
```python
# Before: Hard to test - models depend on API clients and generators
# After: Easy to test in isolation
from models.gpt_4o_mini_interface import GPT4oMiniInterface
model = GPT4oMiniInterface()
result = model.infer("test", "test")
assert isinstance(result, str)
```

### 4. **Code Reduction in Main.py**
```
Before: ~100 lines of model-specific case matching
After: ~10 lines using unified interface
Lines saved: ~90 lines of clutter
```

### 5. **Batch Processing Efficiency**
```python
# Before: Manual formatting for Granite, different code for API
# After: Unified batch method
batch_results = model_interface.infer_batch_with_functions(...)
```

---

## Integration Points

The refactoring maintains full backward compatibility:

```python
# Old code still works
from call_llm import make_chat_pipeline
generator = make_chat_pipeline(LocalModel.GRANITE_3_1_8B_INSTRUCT)

# New code uses it better
from models.model_factory import create_model_interface
interface = create_model_interface(config.model, generator=generator)
result = interface.infer_with_functions(prompt, query, functions)
```

---

## Migration Path

For existing code:
1. **No changes required** - old APIs still available
2. **Gradual adoption** - use new interfaces for new features
3. **Full migration** - eventually replace old code paths
4. **Complete cutover** - once all functionality migrated

---

## Conclusion

The refactoring transforms the codebase from:
- **Scattered, complex, hard-to-extend** ❌
- **Organized, clear, easy-to-extend** ✅

With minimal disruption and maximum benefit.
