## Libraries

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

# Set Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# ## LoRa Fine-Tuning Configuration
# lora_config = LoraConfig(
#     r=16,  
#     lora_alpha=32,  
#     target_modules=["q_proj", "v_proj"],  
#     lora_dropout=0.05,  
#     bias="none",  
# )

# ## Model and Tokenizer
# model_name = "arcee-ai/Llama-3.1-SuperNova-Lite"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply LoRa
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# Dataset Loading from CSV
csv_file = "/Users/mehran/CodeSpaces/Testing/Table_Answering/data/matched_data_2.csv"  # Replace with your actual file path
train_dataset = load_dataset("csv", data_files=csv_file)["train"]

print(train_dataset)