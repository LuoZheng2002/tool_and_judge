
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
# You are a helpful assistant helping rephrasing user requests, while accurately preserving their meaning, including numbers and names if exist. If the request is multilingual, *DON'T* translate English words to other languages and *KEEP* all English words and numbers intact. Do not answer the requirement, just produce another one that is identical in meaning but is phrased differently. Produce ONLY the rephrased requirement, without further thoughts or explanations. Consider the example below:

# USER: 了解在Playstation平台上玩Fortnite通过不同任务和奖杯可获得的奖励

# ASSISTANT: 探索在Playstation平台上玩Fortnite时，通过完成各种任务和获得奖杯可以获得的奖励
# '''
# else:
system_prompt = '''
You are a helpful assistant helping rephrasing user requests, while accurately preserving their meaning, including numbers and names if exist. Do not answer the requirement, just produce another one that is identical in meaning but is phrased differently. Produce ONLY the rephrased requirement, without further thoughts or explanations. Consider the example below:

USER: Can I find the dimensions and properties of a triangle, if it is known that its three sides are 5 units, 4 units and 3 units long?

ASSISTANT: What are the dimensions and properties of a triangle whose three sides are 5, 4 and 3 units long?
'''


def generate_paraphrased_case(question: str) -> str:
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    paraphrased_question = api_inference(ApiModel.GPT_5, input_messages)
    return paraphrased_question



for postfix in postfix_to_generate:
    print(f"Generating paraphrased dataset for postfix: {postfix}")
    original_dataset_path = f'dataset/BFCL_v4_multiple{postfix}.json'
    paraphrased_dataset_path = f'dataset/BFCL_v4_multiple{postfix}_para.json'
    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        original_data = load_json_lines(f)
    paraphrased_data = []
    existing_indices = []
    try:
        with open(paraphrased_dataset_path, 'r', encoding='utf-8') as f:
            paraphrased_data = load_json_lines(f)
            existing_indices = [item['id'] for item in paraphrased_data]
    except FileNotFoundError:
        print(f"No existing paraphrased dataset found at {paraphrased_dataset_path}. A new one will be created.")
    with open(paraphrased_dataset_path, 'w', encoding='utf-8') as f:
        warning_printed = False
        for item in original_data:
            id = item['id']
            if id in existing_indices:
                if not warning_printed:
                    print(f"Warning: Skipping already processed items in {paraphrased_dataset_path}.")
                    warning_printed = True
                continue
            paraphrased_question = generate_paraphrased_case(item['question'][0][0]['content'])
            paraphrased_item = item.copy()
            paraphrased_item['question'][0][0]['content'] = paraphrased_question
            paraphrased_data.append(paraphrased_item)
            f.seek(0)
            f.truncate()
            for n in paraphrased_data:
                f.write(json.dumps(n, ensure_ascii=False) + '\n')
            f.flush()
        # sort
        paraphrased_data = sorted(paraphrased_data, key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf'))
        f.seek(0)
        f.truncate()
        for n in paraphrased_data:
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
        f.flush()
    print(f"Paraphrased dataset with {len(paraphrased_data)} items saved to {paraphrased_dataset_path}.")
    