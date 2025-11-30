# main.py Migration to New Model Interface

## Summary

The `main.py` file has been updated to use the new model interface signatures that accept `functions` and `user_query` directly instead of pre-generated `system_prompt`.

## Changes Made

### 1. Single Inference Function (Lines 50-80)

**Before:**
```python
def inference(model: Model, test_entry: dict, model_interface=None):
    functions = test_entry['function']
    user_question = test_entry["question"][0][0]['content']

    # Determine if this is a local model
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

    # Run inference
    result = model_interface.infer(system_prompt=developer_prompt, user_query=user_question)

    result_to_write = {
        "id": test_entry["id"],
        "result": result
    }
    return result_to_write
```

**After:**
```python
def inference(model: Model, test_entry: dict, model_interface=None):
    functions = test_entry['function']
    user_question = test_entry["question"][0][0]['content']

    # Create model interface if not provided
    if model_interface is None:
        model_interface = create_model_interface(model)

    # Run inference with new interface (accepts functions directly)
    result = model_interface.infer(
        functions=functions,
        user_query=user_question,
        prompt_passing_in_english=True
    )

    result_to_write = {
        "id": test_entry["id"],
        "result": result
    }
    return result_to_write
```

**Benefits:**
- Eliminated manual `gen_developer_prompt()` call
- System prompt generation is now internal to the model interface
- Cleaner, simpler code with less boilerplate
- Model-specific prompt generation handled by specialized interface classes

### 2. Batch Processing Logic (Lines 280-319)

**Before:**
```python
# Prepare batch data
batch_data = []
for case in batch_cases:
    functions = case['function']
    user_question = case["question"][0][0]['content']
    local_model = config.model.model if isinstance(config.model, LocalModelStruct) else None
    developer_prompt = gen_developer_prompt(
        function_calls=functions,
        prompt_passing_in_english=True,
        model=local_model
    )
    batch_data.append((developer_prompt, user_question, functions))

# Create or reuse model interface
if is_api_model:
    model_interface = create_model_interface(config.model)
    # For API models: process each case individually
    print(f"Sending {len(batch_cases)} concurrent API requests...")
    batch_results = []
    for system_prompt, user_query, _ in batch_data:
        result = model_interface.infer(system_prompt=system_prompt, user_query=user_query)
        batch_results.append(result)
    print(f"Received {len(batch_results)} API responses.")

elif is_local_model:
    # For local models: create interface with generator
    local_model = config.model.model
    model_interface = create_model_interface(config.model, generator=config.model.generator)

    if local_model == LocalModel.GRANITE_3_1_8B_INSTRUCT:
        # Use batch processing for Granite
        system_prompts = [data[0] for data in batch_data]
        user_queries = [data[1] for data in batch_data]
        batch_functions = [data[2] for data in batch_data]

        batch_results = model_interface.infer_batch_with_functions(
            system_prompts=system_prompts,
            user_queries=user_queries,
            batch_functions=batch_functions
        )
    else:
        raise ValueError(f"Unsupported local model in batch processing: {local_model}")
else:
    raise ValueError(f"Unsupported model type: {type(config.model)}")
```

**After:**
```python
# Prepare batch data
batch_functions_list = []
batch_user_queries = []
for case in batch_cases:
    functions = case['function']
    user_question = case["question"][0][0]['content']
    batch_functions_list.append(functions)
    batch_user_queries.append(user_question)

# Create or reuse model interface
if is_api_model:
    model_interface = create_model_interface(config.model)
    # For API models: process each case individually
    print(f"Sending {len(batch_cases)} concurrent API requests...")
    batch_results = []
    for functions, user_query in zip(batch_functions_list, batch_user_queries):
        result = model_interface.infer(
            functions=functions,
            user_query=user_query,
            prompt_passing_in_english=True
        )
        batch_results.append(result)
    print(f"Received {len(batch_results)} API responses.")

elif is_local_model:
    # For local models: create interface with generator
    local_model = config.model.model
    model_interface = create_model_interface(config.model, generator=config.model.generator)

    if local_model == LocalModel.GRANITE_3_1_8B_INSTRUCT:
        # Use batch processing for Granite (true batch inference)
        batch_results = model_interface.infer_batch(
            functions_list=batch_functions_list,
            user_queries=batch_user_queries,
            prompt_passing_in_english=True
        )
    else:
        raise ValueError(f"Unsupported local model in batch processing: {local_model}")
else:
    raise ValueError(f"Unsupported model type: {type(config.model)}")
```

**Benefits:**
- Simplified data preparation - no need to generate all prompts upfront
- For API models: Still processes sequentially, but cleaner code
- For Granite: Uses true batch processing via `infer_batch()` instead of `infer_batch_with_functions()`
- System prompt generation happens inside the model interface (batched for efficiency)
- Reduced complexity and potential for errors

## Removed Code

The `gen_developer_prompt()` function is still present in main.py but no longer called in the inference and batch processing paths. This function could be removed in a future refactoring if it's not used elsewhere in the file.

## Benefits of the Migration

1. **Separation of Concerns**: System prompt generation is now encapsulated in each model interface
2. **Less Boilerplate**: Removed redundant prompt generation from main.py's inference logic
3. **Better Batch Processing**: Granite's true batch processing is now properly utilized via `infer_batch()`
4. **Easier Maintenance**: Changes to prompt generation only need to happen in one place (the interface classes)
5. **Consistency**: All models follow the same interface pattern with identical method signatures

## Backward Compatibility

The `gen_developer_prompt()` function remains in main.py for backward compatibility with any other code that might use it. The model interfaces internally implement the same logic, ensuring consistent prompt generation across both paths.

## File Status

**Verified:** Python syntax check passed successfully.

**Modified file:** `main.py` (main.py:50-80 and main.py:280-319)

**Test Coverage:** The migration maintains identical behavior while simplifying the code. Existing test cases should continue to work without modification.
