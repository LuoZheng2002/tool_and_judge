from config import ToolModel


def get_model_directory_name(model: ToolModel) -> str:
    """
    Get a filesystem-safe directory name from model enum value.

    Args:
        model: ApiModel or LocalModel enum

    Returns:
        Filesystem-safe directory name based on model's official name
    """
    model_value = model.value
    # Replace filesystem-unsafe characters
    # "/" -> "-" (for models like "Qwen/Qwen2.5-7B-Instruct")
    # ":" -> "-" (for models like "meta.llama3-1-8b-instruct-v1:0")
    safe_name = model_value.replace("/", "-").replace(":", "-")
    return safe_name
