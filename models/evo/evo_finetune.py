import torch
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_optimizer import APOLLO
from tqdm import tqdm
from huggingface_hub.hf_api import HfFolder
from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Set the Hugging Face token
hf_token = "API_TOKEN"
HfFolder.save_token(hf_token)

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set model config
MODEL_NAME = 'togethercomputer/evo-1-8k-base'
MODEL_CONFIG = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True, revision="1.1_fix")
MODEL_CONFIG.use_cache = True

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision="1.1_fix", trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config=MODEL_CONFIG, trust_remote_code=True).to(DEVICE)

# Load and preprocess the dataset
dataset_name = 'DNA-LLM/experiment_one_viral_genomes_train_set_v2'
sequences = load_dataset(dataset_name)['train']['chunked_seqs']
sequences = [seq.upper() for seq in sequences]

# Tokenize the text
tokenizer.pad_token = tokenizer.eos_token
encodings = tokenizer(sequences, truncation=True, padding=False, max_length=2048)

dataset = Dataset.from_dict({"input_ids": encodings["input_ids"]})
dataset.set_format("torch", columns=["input_ids"])

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=(DEVICE == "cuda"))

# Define optimizer
optimizer = APOLLO(model.parameters(), lr=5e-4)

# Training loop
losses = [0]
steps = [0]

for count, batch in enumerate(tqdm(dataloader), start=1):
    input_ids = batch["input_ids"].to(DEVICE)
    
    # Forward pass
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # Logging
    if count % 100 == 0:
        print(f"Loss: {loss.item():.4f}, Iteration: {count}")
        pd.DataFrame({'step': steps, 'loss': losses}).to_csv('losses.csv', index=False)
    
    if count % 10_000 == 0:
        print(f"Saving model at iteration {count}")
        model.save_pretrained(f"model_{count}", safe_serialization=False)
    
    losses.append(loss.item())
    steps.append(count)

# Final model save
print(f"Saving model at iteration {count}")
model.save_pretrained(f"model_{count}", safe_serialization=False)

# Save loss data
pd.DataFrame({'step': steps, 'loss': losses}).to_csv('losses.csv', index=False)

print(f"Final Loss: {loss.item():.4f}, Iteration: {count}")
