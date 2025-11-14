def count_jsonl_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

# مثال استفاده:
file_path = 'labeled_traces_ALL.jsonl'
print("تعداد خطوط:", count_jsonl_lines(file_path))