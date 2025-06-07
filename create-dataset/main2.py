import json
import random
import os

# ===== 파일 경로 상수 =====
STATUS_MAP_PATH = "data/traffic_status_map.json"
TEMPLATE_PATH = "data/traffic_templates_status.txt"
OUTPUT_DIR = "./dataset"
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.jsonl")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.jsonl")

# ===== 상태 맵 로딩 =====
with open(STATUS_MAP_PATH, "r", encoding="utf-8") as f:
    TRAFFIC_STATUS_MAP = json.load(f)

# ===== 템플릿 로딩 =====
with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    TRAFFIC_TEMPLATES_STATUS = [line.strip() for line in f if line.strip()]

# ===== 좌표 생성 함수 =====
def gen_coord(x_range=(0, 30), y_range=(0, 30)):
    x = random.randint(*x_range)
    y = random.randint(*y_range)
    return x, y

# ===== 중복 방지 세트 =====
used_set = set()

# ===== 샘플 생성 함수 =====
def create_sample():
    while True:
        x, y = gen_coord()
        status_kr = random.choice(list(TRAFFIC_STATUS_MAP.keys()))
        key = (x, y, status_kr)
        if key not in used_set:
            used_set.add(key)
            break

    status_en = TRAFFIC_STATUS_MAP[status_kr]
    template = random.choice(TRAFFIC_TEMPLATES_STATUS)
    sentence = template.format(x=x, y=y, status_kr=status_kr)

    # 프롬프트 및 정답 생성
    prompt = f"### 명령: {sentence}\n### 응답:"
    completion_obj = {
        "action": "change_traffic_status",
        "target": [x, y],
        "status": status_en
    }
    completion_str = json.dumps(completion_obj, ensure_ascii=False)

    return {
        "prompt": prompt,
        "completion": completion_str
    }

# ===== 데이터셋 생성 메인 함수 =====
def create_jsonl_split(train_size=8000, test_size=2000):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = train_size + test_size
    all_samples = [create_sample() for _ in range(total)]

    train_samples = all_samples[:train_size]
    test_samples = all_samples[train_size:]

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(TEST_PATH, "w", encoding="utf-8") as f:
        for s in test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ 학습 데이터: {train_size}개 → {TRAIN_PATH}")
    print(f"✅ 테스트 데이터: {test_size}개 → {TEST_PATH}")

# ===== 학습용 .txt 파일 생성 =====
def create_train_txt(jsonl_path, txt_path):
    with open(jsonl_path, "r", encoding="utf-8") as f_jsonl, open(txt_path, "w", encoding="utf-8") as f_txt:
        for line in f_jsonl:
            sample = json.loads(line)
            f_txt.write(f"{sample['prompt']} {sample['completion']}\n\n")

    print(f"✅ 학습용 텍스트 파일 생성 완료: {txt_path}")

# ===== 메인 실행 =====
if __name__ == "__main__":
    create_jsonl_split(train_size=12000, test_size=2000)
    create_train_txt(TRAIN_PATH, os.path.join(OUTPUT_DIR, "train.txt"))
