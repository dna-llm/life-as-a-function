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

# Select device: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model configuration
model_name = "togethercomputer/evo-1-8k-base"
model_config = AutoConfig.from_pretrained(
    model_name, trust_remote_code=True, revision="1.1_fix"
)
model_config.use_cache = True

# Load tokenizer and model; move model to device and set to training mode
tokenizer = AutoTokenizer.from_pretrained(
    model_name, revision="1.1_fix", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    'evo/model', # location to the directory containing the model files
    config=model_config,
    trust_remote_code=True,
).to(device)

# Load the dataset
test_set = load_dataset('DNA-LLM/DNA-LLM/experiment_one_viral_genomes_test_set_v2')['train']
test_df = pd.DataFrame(test_set)

# Species filtering
species = ['enterovirus c', 'gallivirus a']
sequences = [
    test_df.loc[test_df['species'].str.lower().str.contains(sp), 'chunked_seqs'].iloc[0].upper()
    for sp in species
]

# Tokenize the text
tokenizer.pad_token = tokenizer.eos_token
encodings = tokenizer(sequences, truncation=True, padding=False, max_length=1024)

# Generate sequences
input_ids = torch.tensor(encodings["input_ids"]).to(device)
outputs = model.generate(input_ids, max_length=2048)

# Decode the generated sequences
decoded_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Save results
df = pd.DataFrame({'species': species, 'expected': sequences, 'generated': decoded_sequences})
df.to_csv('generated_sequences.csv', index=False)