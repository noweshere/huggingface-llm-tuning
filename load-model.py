from datasets import load_dataset

# ==========================================
# 2. Dataset & Tokenizer (Hugging Face)
# ==========================================
print("Loading dataset from Hugging Face...")

# We load only the 'train' split, and grab just the first 1% of it 
# to keep memory usage low on the Orin Nano.
hf_dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

# Hugging Face datasets are structured as dictionaries. 
# TinyStories has a 'text' column, so we join the stories into one massive string.
text = "\n".join(hf_dataset["text"])

print(f"Dataset loaded. Total characters in training slice: {len(text)}")

# Unique characters in the text (our vocabulary)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size (unique characters): {vocab_size}")

# Mapping characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

# Convert the giant string into a PyTorch Tensor
print("Tokenizing and converting to tensor...")
data = torch.tensor(encode(text), dtype=torch.long)

# Split into 90% training and 10% validation data
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]
print("Data splitting complete. Ready to train.")


File "/home/nvidia/pytorch/test.py", line 31, in <module>
    hf_dataset = load_dataset("Glint-Research/Fable-5-traces", split="train[:1%]")
  File "/home/nvidia/.local/lib/python3.10/site-packages/datasets/load.py", line 1718, in load_dataset
    builder_instance.download_and_prepare(
  File "/home/nvidia/.local/lib/python3.10/site-packages/datasets/builder.py", line 890, in download_and_prepare
    self._download_and_prepare(
  File "/home/nvidia/.local/lib/python3.10/site-packages/datasets/builder.py", line 951, in _download_and_prepare
    self._prepare_split(split_generator, **prepare_split_kwargs)
  File "/home/nvidia/.local/lib/python3.10/site-packages/datasets/builder.py", line 1683, in _prepare_split
    for job_id, done, content in self._prepare_split_single(
  File "/home/nvidia/.local/lib/python3.10/site-packages/datasets/builder.py", line 1869, in _prepare_split_single
    raise DatasetGenerationError("An error occurred while generating the dataset") from e
datasets.exceptions.DatasetGenerationError: An error occurred while generating the dataset
