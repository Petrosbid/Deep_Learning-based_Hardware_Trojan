import json

file_path = "../jsonl_dataset/labeled_traces_BALANCED.jsonl"
valid, invalid = 0, 0

with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            valid += 1
        except Exception:
            invalid += 1

print(f"✅ خطوط سالم: {valid:,}")
print(f"❌ خطوط خراب: {invalid:,}")
