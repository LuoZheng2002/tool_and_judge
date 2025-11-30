import json

def load_json_lines(file):
    data = []
    for line in file:
        line = line.strip()
        if not line:  # skip empty lines
            continue
        try:
            obj = json.loads(line)
            data.append(obj)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}")
    return data
