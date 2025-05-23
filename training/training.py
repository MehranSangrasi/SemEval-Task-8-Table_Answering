import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format
from datasets import load_dataset
from huggingface_hub import login


# login(
#     token="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",
# )

csv_file = "data\conversations_final.csv"  # Replace with your actual file path
dataset = load_dataset("csv", data_files=csv_file)["train"]

# print(train_dataset[:5])
 
# Hugging Face model id
model_id = "meta-llama/Meta-Llama-3.1-8B" # or `mistralai/Mistral-7B-v0.1`

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings

# # set chat template to OAI chatML, remove if you start from a fine-tuned model
model, tokenizer = setup_chat_format(model, tokenizer)

from peft import LoraConfig

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM", 
)

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="code-llama-3-1-8b-text-to-sql", # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    # push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)


from trl import SFTTrainer

max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model 
trainer.save_model()

#### COMMENT IN TO MERGE PEFT AND BASE MODEL ####
from peft import AutoPeftModelForCausalLM

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)  
# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained(args.output_dir,safe_serialization=True, max_shard_size="2GB")