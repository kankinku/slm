from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return emb[None, :, :]

def apply_rotary(x, rope):
    x1, x2 = x[..., ::2], x[..., 1::2]
    sin, cos = rope[..., ::2], rope[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, D = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        rope = self.rope(T, x.device)
        q = apply_rotary(q, rope)
        k = apply_rotary(k, rope)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(self.dropout(attn_weights), v)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)

        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, 4 * d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class forbus_slm(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))  # Not used with RoPE
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # ✅ Weight Tying
        self.head.weight = self.token_emb.weight

    def forward(self, x):
        B, T = x.shape
        tok = self.token_emb(x)
        x = self.blocks(tok)
        x = self.ln_f(x)
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
model = forbus_slm(vocab_size=tokenizer.vocab_size).to(device)
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
