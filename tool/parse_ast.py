import ast
from config import Model, ApiModel, LocalModel
def recursive_match(value, expected_list):
    """
    Recursively match a value against a list of expected values.
    Handles nested structures (dicts, lists) by comparing structure and values.

    Args:
        value: The value from the model output
        expected_list: A list of possible expected values

    Returns:
        True if the value matches any of the expected values (structurally and by content)
    """
    for expected in expected_list:
        if _matches(value, expected):
            return True
    return False

def _matches(value, expected):
    """
    Helper function to check if value matches expected.
    Handles nested dicts and lists recursively.

    Special handling: if expected is a list but value is NOT a list,
    check if value matches any element in the expected list (unwrap).
    If both are lists, compare them directly (don't unwrap).
    """
    # If both are dicts, compare keys and values recursively
    if isinstance(value, dict) and isinstance(expected, dict):
        if set(value.keys()) != set(expected.keys()):
            return False
        for key in value.keys():
            if not _matches(value[key], expected[key]):
                return False
        return True

    # If both are lists, compare them directly without unwrapping
    if isinstance(value, list) and isinstance(expected, list):
        if len(value) != len(expected):
            return False
        for v, e in zip(value, expected):
            if not _matches(v, e):
                return False
        return True

    # If value is not a list but expected is a list,
    # check if value matches any element in the expected list (unwrap)
    if not isinstance(value, list) and isinstance(expected, list):
        for item in expected:
            if _matches(value, item):
                return True
        return False

    # For other types, use direct equality
    return value == expected

def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        # Convert lowercase "true" and "false" to Python's True and False
        if value.id == "true":
            output = True
        elif value.id == "false":
            output = False
        else:
            output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        # output = tuple(resolve_ast_by_type(v) for v in value.elts) 
        output = [resolve_ast_by_type(v) for v in value.elts] # Zheng: Changed to list to match ground truth
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output

def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}



    
def raw_to_json(model: Model, case_id: str, model_result_raw: str) -> object:
    # print(f"model being decoded: {model}")
    if model == LocalModel.GRANITE_3_1_8B_INSTRUCT:
        # print("Decoding granite model output...")
        import json
        # Parse Granite model's output format: <tool_call>[{...}]
        model_result_raw = model_result_raw.strip()

        # Remove <tool_call> wrapper if present
        if model_result_raw.startswith("<tool_call>"):
            model_result_raw = model_result_raw[len("<tool_call>"):]

        model_result_raw = model_result_raw.strip("`\n ")

        # Add brackets if missing
        if not model_result_raw.startswith("["):
            model_result_raw = "[" + model_result_raw
        if not model_result_raw.endswith("]"):
            model_result_raw = model_result_raw + "]"

        try:
            # Parse the JSON array
            tool_calls = json.loads(model_result_raw)
        except json.JSONDecodeError:
            return f"Failed to decode JSON: Invalid JSON format. Raw string: {model_result_raw}"

        # Convert Granite format to desired format
        extracted = []
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                    func_name = tool_call["name"]
                    func_args = tool_call["arguments"]
                    extracted.append({func_name: func_args})
                else:
                    return f"Failed to decode JSON: Invalid tool call structure. Raw string: {model_result_raw}"
        else:
            return f"Failed to decode JSON: Expected a list of tool calls. Raw string: {model_result_raw}"

        decoded_output = extracted
    else:
        model_result_raw = model_result_raw.strip("`\n ")
        if not model_result_raw.startswith("["):
            model_result_raw = "[" + model_result_raw
        if not model_result_raw.endswith("]"):
            model_result_raw = model_result_raw + "]"
        # We only want to remove wrapping quotes that could have been added by the model.
        cleaned_input = model_result_raw.strip().strip("'")
        try:
            parsed = ast.parse(cleaned_input, mode="eval")
        except SyntaxError:
            return f"Failed to decode AST: Invalid syntax. Raw string: {cleaned_input}"
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                if not isinstance(elem, ast.Call):
                    # raise Exception(f"Expected AST Call node, but got {type(elem)}")
                    return f"Failed to decode AST: Expected AST Call node, but got {type(elem)}. Raw string: {cleaned_input}"
                extracted.append(resolve_ast_call(elem))
        decoded_output = extracted

    return decoded_output


def evaluate_json(
    case_id,
    decoded_output,
    possible_answer,
    func_description,
):
    # print("case id:", case_id)
    # print("decoded output:", decoded_output)
    """Helper method to process a single AST entry."""
    if isinstance(decoded_output, str):
        return {
            "id": case_id,
            "valid": False,
            "error": [decoded_output],
            "error_type": "ast_decoder:decoder_failed",
            "model_result_raw": decoded_output,
            "model_result_decoded": decoded_output,
            "possible_answer": possible_answer,
        }
    
    if len(decoded_output) != 1:
        return {
            "id": case_id,
            "valid": False,
            "error": [f"Expected exactly one AST entry, but got {len(decoded_output)}."],
            "error_type": "ast_checker:invalid_entry_count",
            "model_result_decoded": decoded_output,
            "possible_answer": possible_answer,
        }
    model_result = decoded_output[0]
    possible_answer = possible_answer[0]

    # print("model_result:", model_result)
    # print("possible_answer:", possible_answer)
    assert(len(possible_answer) == 1)
    assert(len(model_result) == 1)
    # print("model_result:", model_result)
    
    # print("possible_answer:", possible_answer)
    # print("possible_answer keys:", possible_answer.keys()   )
    # Extract function name and parameters details
    possible_answer_func_name = next(iter(possible_answer))
    model_result_func_name = next(iter(model_result))
    # print("func_name:", possible_answer_func_name)
    if possible_answer_func_name != model_result_func_name:
        return {
            "id": case_id,
            "valid": False,
            "error": [f"Function name mismatch. Expected {repr(possible_answer_func_name)}, but got {repr(model_result_func_name)}."],
            "error_type": "simple_function_checker:wrong_func_name",
            "model_result_decoded": decoded_output,
            "possible_answer": possible_answer,
        }
    # first see if function name matches

    # param_details = func_description["parameters"]["properties"]
    for func in func_description:
        if func['name'] == possible_answer_func_name:
            # print("func parameters:", func['parameters'].keys())
            required_params = func['parameters']['required']
            break

    # Check for required parameters in model output
    model_params = model_result[possible_answer_func_name]
    for param in required_params:
        if param not in model_params:
            return {
                "id": case_id,
                "valid": False,
                "error": [f"Missing required parameter: {repr(param)}."],
                "error_type": "simple_function_checker:missing_required",
                "model_result_decoded": decoded_output,
                "possible_answer": possible_answer,
            }
        
    possible_answer_params = possible_answer[possible_answer_func_name]
    # print("model_params:", model_params)
    # print(type(model_params))
    # Validate types and values for each parameter in model output
    for param, value in model_params.items():
        if param not in possible_answer_params:
            print("possible_answer keys:", possible_answer.keys())
            print("param: ", param)
            return {
                "id": case_id,
                "valid": False,
                "error": [f"Unexpected parameter: {repr(param)}."],
                "error_type": "simple_function_checker:unexpected_param",
                "model_result_decoded": decoded_output,
                "possible_answer": possible_answer,
            }
        # Check if the value is within the possible answers using recursive matching
        if not recursive_match(value, possible_answer_params[param]):
            return {
                "id": case_id,
                "valid": False,
                "error": [f"Invalid value for parameter {repr(param)}: {repr(value)}. Expected one of {possible_answer_params[param]}."],
                "error_type": "value_error:others",
                "model_result_decoded": decoded_output,
                "possible_answer": possible_answer,
            }
    return {
        "id": case_id,
        "valid": True,
        "model_result_decoded": decoded_output,
        "possible_answer": possible_answer,
    }