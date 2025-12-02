from transformers import AutoTokenizer




if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    print("qwen3-30b-a3b Chat Template:")
    print(tokenizer.chat_template)
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
    # print("Qwen3-Next-80B-A3B-Instruct Chat Template:")
    # print(tokenizer.chat_template)