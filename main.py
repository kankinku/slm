from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing

# 1. 초기 토크나이저 구성
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = Whitespace()

# 2. 학습자 정의
trainer = BpeTrainer(
    vocab_size=3000,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
)

# 3. 학습 실행
tokenizer.train(["./create-dataset/dataset/train.txt"], trainer)

# 4. 후처리 설정 (BOS, EOS)
tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    pair="[BOS] $A [EOS] $B:1 [EOS]:1",
    special_tokens=[
        ("[BOS]", tokenizer.token_to_id("[BOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
    ],
)

# 5. 저장 (Hugging Face 호환 json 파일)
tokenizer.save("tokenizer/kojson-tokenizer.json")
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer/kojson-tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[BOS]",
    eos_token="[EOS]"
)

print(tokenizer.encode("22,9는 우회로개방 상태야", add_special_tokens=True))
import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=524, n_heads=4, n_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        pos = self.pos_embedding(pos)[None, :, :]
        x = self.embedding(x) + pos
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return self.head(x)
from torch.utils.data import Dataset
class PromptCompletionDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        import json
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = f"{data['prompt']} -> {data['completion']}"
                tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=max_len)
                self.samples.append(torch.tensor(tokenized["input_ids"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniGPT(vocab_size=tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
dataset = PromptCompletionDataset("./create-dataset/dataset/train.jsonl", tokenizer)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(device)

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch[:, :-1])
        loss = F.cross_entropy(output.reshape(-1, tokenizer.vocab_size), batch[:, 1:].reshape(-1), ignore_index=tokenizer.pad_token_id)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), f"models/mini_gpt_epoch{epoch+1}.pth")
    
    import torch
import torch.nn.functional as F

def apply_repetition_penalty(logits, input_ids, penalty=1.2):
    # 입력 토큰들에 반복 패널티 적용
    for i in range(logits.size(0)):
        token_ids = input_ids[i].unique()
        logits[i, token_ids] /= penalty
    return logits

def generate(model, input_ids, tokenizer, max_new_tokens=64, top_k=10, penalty=1.2, stop_token="}"):
    model.eval()
    device = input_ids.device
    stop_ids = tokenizer.encode(stop_token, add_special_tokens=False)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)[:, -1, :]  # (B, vocab)
            logits = apply_repetition_penalty(logits, input_ids, penalty=penalty)

            # Top-k sampling
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(values, dim=-1)
            next_token = indices.gather(-1, torch.multinomial(probs, num_samples=1))

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # stop_token 또는 eos_token 발견 시 종료
            if any(t.item() in stop_ids for t in next_token[0]):
                break

            if input_ids.size(1) >= 128:
                break

    return input_ids

# 모델 및 토크나이저 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniGPT(vocab_size=tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load("models/mini_gpt_epoch9.pth", map_location=device))
model.eval()

# 입력 텍스트
test_text = "(22,9)는 '우회로개방' 상태야"
input_ids = tokenizer(test_text, return_tensors="pt")["input_ids"].to(device)

# 텍스트 생성
output_ids = generate(model, input_ids, tokenizer, max_new_tokens=64, top_k=10, penalty=1.2, stop_token="}")
decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# JSON만 추출
start = decoded.find("{")
end = decoded.find("}") + 1
if start != -1 and end > start:
    json_part = decoded[start:end]
    print("⤷ 모델 출력 (JSON):")
    print(json_part)
else:
    print("⤷ 모델 출력:")
    print(decoded)