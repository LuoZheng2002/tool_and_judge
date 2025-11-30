
import json

import re
from call_llm import api_inference
from config import ApiModel
from parse_dataset import load_json_lines


# translated = True
postfix_to_generate = [
    "_zh_partial"
    # "_zh_full"
]


# if translated:
#     system_prompt = '''
# You are a helpful assistant that replaces words with synonyms of similar meaning while maintaining semantic correctness. Your task is to process word by word and replace each word with a synonym if possible.

# IMPORTANT RULES:
# 1. Replace ONLY non-English words with appropriate synonyms
# 2. KEEP all English words and numbers unchanged
# 3. Maintain the semantic meaning and grammatical structure
# 4. Do NOT perform general paraphrasing, only synonym replacement
# 5. Process word by word, not phrase by phrase
# 6. If a word has no suitable synonym or is a proper noun, keep it unchanged

# Produce ONLY the modified text with synonyms, without further thoughts or explanations. Consider the example below:

# USER: 了解在Playstation平台上玩Fortnite通过不同任务和奖杯可获得的奖励

# ASSISTANT: 理解在Playstation平台上玩Fortnite经过各类工作和奖项可获得的回报
# '''
# else:
system_prompt = '''
You are a helpful assistant that replaces words with synonyms of similar meaning while maintaining semantic correctness. Your task is to process word by word and replace each word with a synonym if possible.

IMPORTANT RULES:
1. Replace words with appropriate synonyms
2. Maintain the semantic meaning and grammatical structure
3. Do NOT perform general paraphrasing, only synonym replacement
4. Process word by word, not phrase by phrase
5. If a word has no suitable synonym or is a proper noun, keep it unchanged

Produce ONLY the modified text with synonyms, without further thoughts or explanations. Consider the example below:

USER: Can I find the dimensions and properties of a triangle, if it is known that its three sides are 5 units, 4 units and 3 units long?

ASSISTANT: Can I discover the measurements and characteristics of a triangle, if it is known that its three sides are 5 units, 4 units and 3 units long?
'''

def generate_synonym_case(question: str) -> str:
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    synonym_question = api_inference(ApiModel.GPT_5, input_messages)
    return synonym_question



for postfix in postfix_to_generate:
    print(f"Generating synonym dataset for postfix: {postfix}")
    original_dataset_path = f'dataset/BFCL_v4_multiple{postfix}.json'
    synonym_dataset_path = f'dataset/BFCL_v4_multiple{postfix}_syno.json'
    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        original_data = load_json_lines(f)
    synonym_data = []
    existing_indices = []
    try:
        with open(synonym_dataset_path, 'r', encoding='utf-8') as f:
            synonym_data = load_json_lines(f)
            existing_indices = [item['id'] for item in synonym_data]
    except FileNotFoundError:
        print(f"No existing synonym dataset found at {synonym_dataset_path}. A new one will be created.")
    with open(synonym_dataset_path, 'w', encoding='utf-8') as f:
        warning_printed = False
        for item in original_data:
            id = item['id']
            if id in existing_indices:
                if not warning_printed:
                    print(f"Warning: Skipping already processed items in {synonym_dataset_path}.")
                    warning_printed = True
                continue
            synonym_question = generate_synonym_case(item['question'][0][0]['content'])
            synonym_item = item.copy()
            synonym_item['question'][0][0]['content'] = synonym_question
            synonym_data.append(synonym_item)
            f.seek(0)
            f.truncate()
            for n in synonym_data:
                f.write(json.dumps(n, ensure_ascii=False) + '\n')
            f.flush()
        # sort
        synonym_data = sorted(synonym_data, key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf'))
        f.seek(0)
        f.truncate()
        for n in synonym_data:
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
        f.flush()
    print(f"Synonym dataset with {len(synonym_data)} items saved to {synonym_dataset_path}.")
