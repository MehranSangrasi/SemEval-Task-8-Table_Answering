
from torch import __version__; from packaging.version import Version as V
xformers = "xformers==0.0.27" if V(__version__) < V("2.4.0") else "xformers"

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported



# Load model
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

# Prepare model for PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)
print(model.print_trainable_parameters())


# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="chatml",
#     mapping={"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}
# )

# def apply_template(examples):
#     messages = examples["conversations"]
#     text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
#     return {"text": text}


# def preprocess_text(examples):
#     inputs = [f'Question: {q} \n Dataset Columns: {d}' for q,d in zip(examples['question'], examples['columns'])]
#     targets = [a for a in examples['query']]
#     model_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=512)
#     labels = tokenizer(targets, padding='max_length', truncation=True, max_length=512)
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    print(text)
    return {"text": text}




# dataset = load_dataset("mlabonne/FineTome-100k", split="train")

csv_file = "/home/mehran1/projects/def-cjhuofw-ab/mehran1/SemEval/data/conversations.csv"  # Replace with your actual file path
train_dataset = load_dataset("csv", data_files=csv_file)["train"]

dataset = train_dataset.map(apply_template, batched=False)




trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

trainer.train()


model.save_pretrained("/home/mehran1/projects/def-cjhuofw-ab/mehran1/SemEval/training")
tokenizer.save_pretrained("/home/mehran1/projects/def-cjhuofw-ab/mehran1/SemEval/training")