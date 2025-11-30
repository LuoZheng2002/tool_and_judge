from openai import OpenAI
import json
from dotenv import load_dotenv
import os
from enum import Enum, auto
from config import Language
import re
from parse_dataset import load_json_lines
# Load API keys from .env file
load_dotenv(dotenv_path=".env")


class TranslateOption(Enum):
    FULLY_TRANSLATED = auto()
    PARTIALLY_TRANSLATED = auto()

class TranslateConfig:
    def __init__(self, language: Language, option: TranslateOption):
        self.language = language
        self.option = option

# language_to_translate: list[Language] = [Language.CHINESE]

translate_configs: list[TranslateConfig] = [
    TranslateConfig(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED),
    TranslateConfig(language=Language.CHINESE, option=TranslateOption.PARTIALLY_TRANSLATED),
    TranslateConfig(language=Language.HINDI, option=TranslateOption.FULLY_TRANSLATED),
    TranslateConfig(language=Language.HINDI, option=TranslateOption.PARTIALLY_TRANSLATED),
]

# input_postfix = "_noisy"
input_postfix = ""

for config in translate_configs:
    print(f"Translating dataset for language: {config.language}, option: {config.option}")
    # === Choose which model to use ===
    # Options: "deepseek" or "openai"
    # MODEL_PROVIDER = "deepseek"  # change this to "openai" to switch

    # === Model and client configuration ===
    match config.language:
        case Language.CHINESE:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = "https://api.deepseek.com"
            model_name = "deepseek-chat"
        case Language.HINDI:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1"
            model_name = "gpt-4o-mini"
        case _:
            raise ValueError("MODEL_PROVIDER must be either 'deepseek' or 'openai'")
    
    match config.language:
        case Language.CHINESE:
            translate_mode_postfix = "_zh"
        case Language.HINDI:
            translate_mode_postfix = "_hi"
    match config.option:
        case TranslateOption.FULLY_TRANSLATED:
            translate_mode_postfix += "_full"
        case TranslateOption.PARTIALLY_TRANSLATED:
            translate_mode_postfix += "_partial"
    output_path = f"dataset/BFCL_v4_multiple{input_postfix}{translate_mode_postfix}.json"
    # Initialize the client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # === Load input file ===
    with open(f"dataset/BFCL_v4_multiple{input_postfix}.json", "r", encoding="utf-8") as f:
        dataset = load_json_lines(f)
    with open("dataset/possible_answer/BFCL_v4_multiple.json", "r", encoding="utf-8") as f:
        possible_answers = load_json_lines(f)
    # === Process each line ===
    existing_indices = []
    translated_lines = []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            translated_lines = load_json_lines(f)
            existing_indices = [item["id"] for item in translated_lines]
    except FileNotFoundError:
        print(f"No existing translated dataset found at {output_path}. A new one will be created.")
    with open(output_path, "w", encoding="utf-8") as f_out:
        warning_printed = False
        for dataset_line in dataset:
            id = dataset_line["id"]
            if id in existing_indices:
                if not warning_printed:
                    print(f"Warning: Skipping already processed items in {output_path}.")
                    warning_printed = True
                continue
            # === Translation system message ===
            dataset_question = dataset_line["question"][0][0]["content"]
            match (config.language, config.option):
                case (Language.CHINESE, TranslateOption.FULLY_TRANSLATED):
                    system_message = {
                        "role": "system",
                        "content": "你是一个翻译助手，请将用户提的问题翻译成中文，不要回答问题，不要换行。"
                    }
                    user_message = {"role": "user", "content": dataset_question}
                case (Language.CHINESE, TranslateOption.PARTIALLY_TRANSLATED):
                    possible_answer = next((ans for ans in possible_answers if ans["id"] == id), None)
                    # retrieve the first value of the possible answer dict

                    assert possible_answer is not None, f"Possible answer not found for id {id}"
                    possible_answer = possible_answer['ground_truth'][0]
                    possible_answer = next(iter(possible_answer.values()))
                    possible_answer = json.dumps(possible_answer, ensure_ascii=False)
                    system_message = {
                        "role": "system",
                        "content": '''
你是一个翻译助手，用户将输入两行内容，第一行是需要翻译的句子，第二行是一个JSON字符串，包含了一些词语。如果这些词语出现在需要翻译的句子中，请保持这些词语不变，只翻译其他部分。请将第一行用户提的问题翻译成中文，不要回答问题，不要换行。

以下是一个例子：
用户输入：
Can you calculate the displacement of a car moving at an initial speed of 20 m/s and then accelerates at 10 m/s^2 for 5 seconds? (assuming a straight line motion)
{"initial_speed": [20], "acceleration": [10], "time": [5], "rounding": ["", 2]}
输出：
你能计算一辆汽车以initial speed 20米/秒行驶，然后以10米/秒²的acceleration加速5秒钟后的位移吗？（假设是直线运动）

注意到initial speed和acceleration不翻译，因为它们出现在了JSON字符串中。
'''     
                    }
                    user_message = {"role": "user", "content": f'''
{dataset_question}
{possible_answer}
                                    '''}
                    # print("user message:")
                    # print(user_message["content"])
                    # exit(1)
                case (Language.HINDI, TranslateOption.FULLY_TRANSLATED):
                    system_message = {
                        "role": "system",
                        "content": "You are a translation assistant. Please translate the user's question into Hindi. Do not answer the question. Do not add line breaks."
                    }
                    user_message = {"role": "user", "content": dataset_question}
                case (Language.HINDI, TranslateOption.PARTIALLY_TRANSLATED):
                    possible_answer = next((ans for ans in possible_answers if ans["id"] == id), None)
                    # retrieve the first value of the possible answer dict

                    assert possible_answer is not None, f"Possible answer not found for id {id}"
                    possible_answer = possible_answer['ground_truth'][0]
                    possible_answer = next(iter(possible_answer.values()))
                    possible_answer = json.dumps(possible_answer, ensure_ascii=False)
                    system_message = {
                        "role": "system",
                        "content": '''
You are a translation assistant. The user will input two lines:
the first line is a sentence to be translated,
and the second line is a JSON string containing some words.

If any of those words appear in the sentence, keep them unchanged (do not translate them), and translate all other parts.

Translate the first line (the user’s sentence) into Hindi, without answering the question and without adding a line break.

Below is an example:

User input:
Can you calculate the displacement of a car moving at an initial speed of 20 m/s and then accelerates at 10 m/s^2 for 5 seconds? (assuming a straight line motion)
{"initial_speed": [20], "acceleration": [10], "time": [5], "rounding": ["", 2]}

Output:
क्या आप गणना कर सकते हैं कि एक कार जो initial speed 20 मी/सेकंड की गति से चल रही है और फिर 10 मी/सेकंड² की acceleration से 5 सेकंड तक तेज़ होती है, उसका विस्थापन कितना होगा? (मान लीजिए यह सीधी रेखा में चल रही है)

Note that initial speed and acceleration are kept untranslated because they appear in the JSON string.
'''     
                    }
                    user_message = {"role": "user", "content": f'''
{dataset_question}
{possible_answer}
                                    '''}

            # === Prepare user message and call API ===
            response = client.chat.completions.create(
                model=model_name,
                messages=[system_message, user_message],
            )

            translated = response.choices[0].message.content.strip()

            # Update the line content
            translated_line = dataset_line.copy()
            translated_line["question"][0][0]["content"] = translated
            translated_lines.append(translated_line)
            f_out.seek(0)
            f_out.truncate()
            for t_line in translated_lines:
                f_out.write(json.dumps(t_line, ensure_ascii=False) + "\n")
            f_out.flush()
        # sort the lines
        translated_lines = sorted(translated_lines, key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf'))
        f_out.seek(0)
        f_out.truncate()
        for t_line in translated_lines:
            f_out.write(json.dumps(t_line, ensure_ascii=False) + "\n")
        f_out.flush()
    print(f"\n✅ Translation complete! Output saved to: {output_path}")

    # # === Save output file ===
    # output_filename = f"BFCL_v4_multiple_zh_q_{MODEL_PROVIDER}.json"
    # with open(output_filename, "w", encoding="utf-8") as f:
    #     for dataset_line in dataset:
    #         f.write(json.dumps(dataset_line, ensure_ascii=False) + "\n")

    # print(f"\n✅ Translation complete! Output saved to: {output_filename}")
