
import json
import os

def process_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Processing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    truncated_count = 0
    for line in lines:
        line = line.strip()
        if not line: continue
        try:
            data = json.loads(line)
            # messages[1] is assistant response
            if "messages" in data and len(data["messages"]) > 1 and data["messages"][1]["role"] == "assistant":
                content = data["messages"][1]["content"]
                if len(content) > 4000:
                    data["messages"][1]["content"] = content[:4000]
                    truncated_count += 1
            new_lines.append(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            print(f"Error parsing line: {e}")
            continue
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_lines) + "\n")
    print(f"Processed {filepath}: {len(lines)} items. Truncated {truncated_count} items.")

if __name__ == "__main__":
    process_file("data/train.jsonl")
    process_file("data/valid.jsonl")
