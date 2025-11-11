import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Word count: {len(raw_text)}")
print(raw_text[:99])

result = re.split(r'(\s)', raw_text)
result = [item.strip() for item in result if item.strip()]

print(result[:49])
