import torch
import json
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset

# ==========================================
# 1. Hyperparameters (Tuned for Orin Nano)
# ==========================================
batch_size = 32      # How many independent sequences will we process in parallel?
block_size = 256      # What is the maximum context length for predictions?
max_iters = 2000     # Number of training steps
eval_interval = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 256         # Embedding dimension
n_head = 8           # Number of attention heads
n_layer = 12          # Number of transformer layers
dropout = 0.1

print(f"Using device: {device}")
torch.manual_seed(1337)

# ==========================================
# 2. Dataset & Tokenizer
# ==========================================
print("Loading dataset from Hugging Face...")

# We load only the 'train' split, and grab just the first 1% of it 
# to keep memory usage low on the Orin Nano.
hf_dataset = load_dataset("json", data_files="https://huggingface.co/datasets/Glint-Research/Fable-5-traces/resolve/main/fable5_cot_merged.jsonl", split="train[:50%]")

# print(hf_dataset)
# print(hf_dataset.column_names)
# print(hf_dataset[0])

# Hugging Face datasets are structured as dictionaries. 
text = "\n\n".join(
    "\n".join([
        ex["context"],
        ex["cot"],
        json.dumps(ex["output"], ensure_ascii=False),
        ex["completion"],
    ])
    for ex in hf_dataset
)

print(f"Dataset loaded. Total characters in training slice: {len(text)}")

# Unique characters in the text (our vocabulary)
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mapping characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

# Data loading function
def get_batch(split):
    data_src = train_data if split == 'train' else val_data
    # Prevent index out of bounds
    max_idx = len(data_src) - block_size - 1
    if max_idx <= 0:
        # Fallback if text is too short
        ix = torch.zeros((batch_size,), dtype=torch.long)
    else:
        ix = torch.randint(max_idx, (batch_size,))
    x = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==========================================
# 3. Transformer Components
# ==========================================
class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C**-0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        
        # Perform the weighted aggregation of the values
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# ==========================================
# 4. The Language Model Network
# ==========================================
class CustomLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=20):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                probs = F.softmax(values, dim=-1)
                sample = torch.multinomial(probs, 1)
                idx_next = indices.gather(-1, sample)
            else:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, 1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# Instantiate the model
model = CustomLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ==========================================
# 5. Training Loop
# ==========================================
print("Starting training Loop...")
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Clear CUDA cache after training
torch.cuda.empty_cache()

# ==========================================
# 6. Generate text from the trained model
# ==========================================
print("\n--- Generating Text ---")
prompt = "USER: Hello! \n ASSISTANT: "
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated = model.generate(context, 300)[0].tolist()
print(decode(generated))
