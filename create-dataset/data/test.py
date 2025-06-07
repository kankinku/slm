import re

input_path = "data/traffic_templates_status.txt"
output_path = "data/traffic_templates_status.txt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []
for line in lines:
    line = re.sub(r"\{ ?x ?\}", "{x}", line)
    line = re.sub(r"\{ ?y ?\}", "{y}", line)
    line = re.sub(r"\{ ?status_kr ?\}", "{status_kr}", line)
    cleaned_lines.append(line.strip())

with open(output_path, "w", encoding="utf-8") as f:
    for line in cleaned_lines:
        if line:
            f.write(line + "\n")

print(f"✅ 템플릿 파일 정제 완료: {output_path}")
