import json
import re
from parse_dataset import load_json_lines
from call_llm import api_inference
from config import ApiModel


def revise_single_pair_with_llm(paraphrased_content, original_content):
    """Use LLM to restore English words from original to paraphrased."""
    system_prompt = '''You will receive two lines of text:
1. The first line is the paraphrased text (may have some English words translated to other languages)
2. The second line is the original text (contains the correct English words)

Your task:
- If you see any English words from the second line that do not appear in the first line, find the corresponding word in the first line and replace it with the exact English word from the second line.
- Only output the revised first line.
- Do not add any explanations or extra text.

Example:
Fist line: 寻找一份少于500卡路里的健康午餐菜谱，需包含鸡肉和蘑菇。
Second line: 找一份低于500卡路里的健康午餐食谱，需包含chicken和mushrooms。
Output: 寻找一份少于500卡路里的健康午餐菜谱，需包含chicken和mushrooms。

Notice how output is based on the first line, but with "chicken" and "mushrooms" restored from the second line.
'''

    user_prompt = f"""{paraphrased_content}
{original_content}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = api_inference(ApiModel.DEEPSEEK_CHAT, messages)
        return response.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return paraphrased_content


def revise_noise(original_file, paraphrased_file, output_file):
    """
    Read original and paraphrased datasets, compare their content,
    and fix mistranslated English words/numbers in the paraphrased dataset.

    Args:
        original_file: Path to the original dataset file
        paraphrased_file: Path to the paraphrased dataset file
        output_file: Path to the output revised paraphrased dataset file    
    """
    print(f"Loading original dataset from {original_file}...")
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = load_json_lines(f)

    print(f"Loading paraphrased dataset from {paraphrased_file}...")
    with open(paraphrased_file, 'r', encoding='utf-8') as f:
        paraphrased_data = load_json_lines(f)

    # Create mapping by ID for quick lookup
    original_by_id = {item['id']: item for item in original_data}

    # Find discrepancies and fix them
    revised_data = []
    discrepancies_fixed = 0
    existing_indices = []

    # Load existing revised data if it exists
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            revised_data = load_json_lines(f)
            existing_indices = [item['id'] for item in revised_data]
    except FileNotFoundError:
        print(f"No existing revised dataset found at {output_file}. A new one will be created.")

    print(f"\nProcessing and revising dataset...")
    with open(output_file, 'w', encoding='utf-8') as f:
        warning_printed = False
        for i, paraphrased_item in enumerate(paraphrased_data):
            item_id = paraphrased_item['id']

            # Skip already processed items
            if item_id in existing_indices:
                if not warning_printed:
                    print(f"Warning: Skipping already processed items in {output_file}.")
                    warning_printed = True
                continue

            # Get the corresponding original item
            if item_id not in original_by_id:
                print(f"Warning: {item_id} not found in original dataset, keeping paraphrased version")
                revised_data.append(paraphrased_item)
                f.seek(0)
                f.truncate()
                for n in revised_data:
                    f.write(json.dumps(n, ensure_ascii=False) + '\n')
                f.flush()
                continue

            original_item = original_by_id[item_id]

            # Extract content from the nested structure
            try:
                original_content = original_item['question'][0][0]['content']
                paraphrased_content = paraphrased_item['question'][0][0]['content']
            except (KeyError, IndexError, TypeError):
                print(f"Warning: Could not extract content from {item_id}, keeping paraphrased version")
                revised_data.append(paraphrased_item)
                f.seek(0)
                f.truncate()
                for n in revised_data:
                    f.write(json.dumps(n, ensure_ascii=False) + '\n')
                f.flush()
                continue

            # Use LLM to revise the paraphrased content
            print(f"\n[{i+1}] Processing {item_id}")
            print(f"    Paraphrased: {paraphrased_content}")
            print(f"    Original:    {original_content}")

            revised_content = revise_single_pair_with_llm(paraphrased_content, original_content)

            print(f"    Revised:     {revised_content}")

            # Update the paraphrased item with revised content
            revised_item = paraphrased_item.copy()
            revised_item['question'][0][0]['content'] = revised_content
            revised_data.append(revised_item)
            discrepancies_fixed += 1

            # Flush to file after each item
            f.seek(0)
            f.truncate()
            for n in revised_data:
                f.write(json.dumps(n, ensure_ascii=False) + '\n')
            f.flush()

        # Sort the revised data by ID
        revised_data = sorted(
            revised_data,
            key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf')
        )
        f.seek(0)
        f.truncate()
        for n in revised_data:
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
        f.flush()

    print(f"\nRevision complete!")
    print(f"  Total items processed: {len(revised_data)}")
    print(f"  Items revised: {discrepancies_fixed}")


if __name__ == "__main__":
    # Example usage
    original_file = "dataset/BFCL_v4_multiple_zh_partial.json"
    paraphrased_file = "dataset/BFCL_v4_multiple_zh_partial_synonym.json"
    output_file = "dataset/BFCL_v4_multiple_zh_partial_synonym_revised.json"

    revise_noise(original_file, paraphrased_file, output_file)
