"""
Partially translate BFCL questions to Chinese while preserving keywords from ground truth.
Supports two modes:
- Thinking mode: Uses deepseek-reasoner with built-in reasoning
- Non-thinking mode: Uses deepseek-chat with explicit reasoning and \\box{} output
"""

import argparse
import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from parse_dataset import load_json_lines

# Load API keys from .env file
load_dotenv(dotenv_path=".env")


def create_thinking_mode_prompt(
    question_content: str,
    functions: list[dict],
    ground_truth: list[dict]
) -> str:
    """
    Create a prompt for Deepseek reasoner (thinking mode).
    The model will reason internally and output only the final translation.
    """
    prompt = f"""## 任务背景

我们正在构建一个测试数据集，用于评估AI模型用中文进行推理并调用函数的能力。原始问题是英文的，我们需要将其翻译成中文。

**关键挑战**：当问题被完全翻译成中文后，模型往往会用中文填写函数参数（如把"New York"翻译成"纽约"），导致与预期的英文ground truth不匹配。因此，我们需要进行**部分翻译**——将问题主体翻译成中文，但保留那些会出现在函数调用参数中的关键信息不变。

## 原始问题（英文）
{question_content}

## 可用的函数定义
{json.dumps(functions, ensure_ascii=False, indent=2)}

## 期望的函数调用（Ground Truth）
{json.dumps(ground_truth, ensure_ascii=False, indent=2)}

## 你的任务

将上述英文问题翻译成中文，但要确保翻译后的问题仍能引导AI模型生成与ground truth完全匹配的函数调用。

## 思考过程中请验证

1. **分析ground truth**：仔细查看ground truth中的函数名和参数值，识别哪些信息必须保持原样才能让模型正确提取
2. **确定保留策略**：决定哪些词语、数字、名称、日期等需要保留英文原文
3. **进行翻译**：将问题翻译成自然流畅的中文，同时保留必要的关键信息
4. **验证翻译结果**：想象一个AI模型看到你翻译后的问题和函数定义，它是否能够：
   - 选择正确的函数？
   - 提取出与ground truth匹配的参数值？
5. **如有必要，调整翻译**：如果验证发现问题，修改翻译以确保正确性

## 输出要求

只输出翻译后的问题，不要包含任何解释、原文、思考过程或其他内容。"""

    return prompt


def create_non_thinking_mode_prompt(
    question_content: str,
    functions: list[dict],
    ground_truth: list[dict]
) -> str:
    """
    Create a prompt for non-thinking mode (deepseek-chat).
    The model should reason explicitly in its response and output the final answer in \\box{}.
    """
    prompt = f"""## 任务背景

我们正在构建一个测试数据集，用于评估AI模型用中文进行推理并调用函数的能力。原始问题是英文的，我们需要将其翻译成中文。

**关键挑战**：当问题被完全翻译成中文后，模型往往会用中文填写函数参数（如把"New York"翻译成"纽约"），导致与预期的英文ground truth不匹配。因此，我们需要进行**部分翻译**——将问题主体翻译成中文，但保留那些会出现在函数调用参数中的关键信息不变。

## 原始问题（英文）
{question_content}

## 可用的函数定义
{json.dumps(functions, ensure_ascii=False, indent=2)}

## 期望的函数调用（Ground Truth）
{json.dumps(ground_truth, ensure_ascii=False, indent=2)}

## 你的任务

将上述英文问题翻译成中文，但要确保翻译后的问题仍能引导AI模型生成与ground truth完全匹配的函数调用。

## 请按以下步骤进行推理（请在回复中展示你的思考过程）

1. **分析ground truth**：仔细查看ground truth中的函数名和参数值，识别哪些信息必须保持原样才能让模型正确提取。列出这些关键信息。

2. **确定保留策略**：决定哪些词语、数字、名称、日期等需要保留英文原文。解释你的决定。

3. **进行翻译**：将问题翻译成自然流畅的中文，同时保留必要的关键信息。展示你的初步翻译。

4. **验证翻译结果**：想象一个AI模型看到你翻译后的问题和函数定义，验证它是否能够：
   - 选择正确的函数？
   - 提取出与ground truth匹配的参数值？

5. **如有必要，调整翻译**：如果验证发现问题，修改翻译以确保正确性。

## 输出要求

在完成上述推理后，将最终的翻译结果放在 \\box{{}} 中。例如：

\\box{{这是翻译后的问题}}

注意：\\box{{}} 中只包含翻译后的问题，不要包含任何其他内容。"""

    return prompt


def extract_box_content(text: str) -> str | None:
    """
    Extract content from \\box{} in the response.
    Handles nested braces properly.
    """
    # Try to find \box{...} pattern
    # First, try simple regex for non-nested cases
    simple_match = re.search(r'\\box\{([^{}]*)\}', text)
    if simple_match:
        return simple_match.group(1).strip()

    # For nested braces, use a more sophisticated approach
    box_start = text.find('\\box{')
    if box_start == -1:
        # Try alternative formats
        box_start = text.find('\\box {')
        if box_start != -1:
            box_start += 6  # len('\\box {')
        else:
            return None
    else:
        box_start += 5  # len('\\box{')

    # Count braces to find matching closing brace
    brace_count = 1
    pos = box_start
    while pos < len(text) and brace_count > 0:
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
        pos += 1

    if brace_count == 0:
        return text[box_start:pos-1].strip()

    return None


def translate_with_deepseek(
    client: OpenAI,
    prompt: str,
    model: str,
    thinking_mode: bool
) -> tuple[str | None, str]:
    """
    Use Deepseek to translate the question.

    Args:
        client: OpenAI client
        prompt: The translation prompt
        model: Model name to use
        thinking_mode: If True, use thinking mode (deepseek-reasoner)
                      If False, use non-thinking mode with explicit reasoning

    Returns:
        tuple: (translated_text, reasoning_content)
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        message = response.choices[0].message
        raw_content = message.content.strip() if message.content else ""

        # Get reasoning content if available (for deepseek-reasoner)
        reasoning_content = ""
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning_content = message.reasoning_content

        if thinking_mode:
            # In thinking mode, the output is directly the translation
            translated_text = raw_content
        else:
            # In non-thinking mode, extract from \box{}
            translated_text = extract_box_content(raw_content)
            if translated_text is None:
                print(f"  Warning: Could not extract \\box{{}} content from response")
                print(f"  Raw response: {raw_content[:500]}...")
                # Fallback: try to use the last line or paragraph as the translation
                lines = [l.strip() for l in raw_content.split('\n') if l.strip()]
                if lines:
                    translated_text = lines[-1]
                else:
                    translated_text = None
            # Store the full response as reasoning for non-thinking mode
            reasoning_content = raw_content

        return translated_text, reasoning_content

    except Exception as e:
        print(f"Error calling Deepseek API: {e}")
        return None, ""


def main():
    # === Parse command line arguments ===
    parser = argparse.ArgumentParser(
        description="Partially translate BFCL questions to Chinese"
    )
    parser.add_argument(
        "--thinking-mode",
        action="store_true",
        default=True,
        help="Use thinking mode with deepseek-reasoner (default: True)"
    )
    parser.add_argument(
        "--no-thinking-mode",
        action="store_true",
        help="Use non-thinking mode with deepseek-chat and explicit reasoning"
    )
    parser.add_argument(
        "--input",
        default="dataset/BFCL_v4_multiple.json",
        help="Input dataset file"
    )
    parser.add_argument(
        "--ground-truth",
        default="dataset/possible_answer/BFCL_v4_multiple.json",
        help="Ground truth file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (default: auto-generated based on mode)"
    )

    args = parser.parse_args()

    # Determine mode
    thinking_mode = not args.no_thinking_mode

    # === Configuration ===
    input_file = args.input
    ground_truth_file = args.ground_truth

    # if args.output:
    #     output_file = args.output
    # else:
    #     # Auto-generate output filename based on mode
    #     if thinking_mode:
    #         output_file = "dataset/BFCL_v4_multiple_zh_partial_thinking.json"
    #     else:
    #         output_file = "dataset/BFCL_v4_multiple_zh_partial_non_thinking.json"
    output_file = "dataset/BFCL_v4_multiple_zh_partial.json"

    # Deepseek configuration
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY not found in .env")

    base_url = "https://api.deepseek.com"

    if thinking_mode:
        model_name = "deepseek-reasoner"  # Thinking mode model
        print("Using THINKING MODE with deepseek-reasoner")
    else:
        model_name = "deepseek-chat"  # Non-thinking mode model
        print("Using NON-THINKING MODE with deepseek-chat")

    # Initialize the client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # === Load input files ===
    print(f"Loading dataset from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = load_json_lines(f)

    print(f"Loading ground truth from {ground_truth_file}...")
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truths = load_json_lines(f)

    # Create a mapping of id to ground truth
    gt_map = {gt['id']: gt['ground_truth'] for gt in ground_truths}

    # === Load existing translations (for resumption) ===
    existing_indices = []
    translated_lines = []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            translated_lines = load_json_lines(f)
            existing_indices = [item["id"] for item in translated_lines]
        print(f"Found {len(existing_indices)} existing translations in {output_file}")
    except FileNotFoundError:
        print(f"No existing translated dataset found at {output_file}. A new one will be created.")

    # === Process each question ===
    print(f"\nProcessing {len(dataset)} questions...")
    warning_printed = False

    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, dataset_line in enumerate(dataset):
            item_id = dataset_line["id"]

            # Skip already processed items
            if item_id in existing_indices:
                if not warning_printed:
                    print(f"Warning: Skipping already processed items.")
                    warning_printed = True
                continue

            # Get question content
            question_content = dataset_line["question"][0][0]["content"]
            functions = dataset_line["function"]

            # Get ground truth
            if item_id not in gt_map:
                print(f"Warning: No ground truth found for {item_id}, skipping...")
                continue

            ground_truth = gt_map[item_id]

            # Create translation prompt based on mode
            if thinking_mode:
                prompt = create_thinking_mode_prompt(
                    question_content,
                    functions,
                    ground_truth
                )
            else:
                prompt = create_non_thinking_mode_prompt(
                    question_content,
                    functions,
                    ground_truth
                )

            # Translate using Deepseek
            print(f"\n[{i+1}/{len(dataset)}] Translating {item_id}...")
            translated_content, reasoning = translate_with_deepseek(
                client, prompt, model_name, thinking_mode
            )

            if translated_content:
                # Create translated item
                translated_line = dataset_line.copy()
                translated_line["question"] = [[{
                    "role": "user",
                    "content": translated_content
                }]]
                translated_lines.append(translated_line)

                print(f"  Original: {question_content[:80]}...")
                print(f"  Translated: {translated_content[:80]}...")

                # Save progress after each translation
                f_out.seek(0)
                f_out.truncate()

                # Sort by id before saving
                sorted_lines = sorted(
                    translated_lines,
                    key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf')
                )
                for t_line in sorted_lines:
                    f_out.write(json.dumps(t_line, ensure_ascii=False) + "\n")
                f_out.flush()
            else:
                print(f"  Warning: Translation failed for {item_id}")
                exit(1)

        # Final sort and save
        sorted_lines = sorted(
            translated_lines,
            key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf')
        )
        f_out.seek(0)
        f_out.truncate()
        for t_line in sorted_lines:
            f_out.write(json.dumps(t_line, ensure_ascii=False) + "\n")
        f_out.flush()

    print(f"\n✅ Translation complete! Output saved to: {output_file}")
    print(f"Total items translated: {len(translated_lines)}")


if __name__ == "__main__":
    main()
