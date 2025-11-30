from transformers import AutoTokenizer




if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-h-tiny")
    print("Granite 4.0-h-tiny Chat Template:")
    print(tokenizer.chat_template)
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
    # print("Qwen3-Next-80B-A3B-Instruct Chat Template:")
    # print(tokenizer.chat_template)