import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from datasets import load_dataset

# Load the model
model_id = "./code-llama-3-1-8b-text-to-sql"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load the dataset
csv_file = "data/conversations_final.csv"  # Replace with your actual file path
dataset = load_dataset("csv", data_files=csv_file)["train"]

# Extract the last two rows
last_two_rows = dataset[-2:]

# Perform inference for the last two rows
for idx, row in enumerate(last_two_rows):
    # Extract user content as the prompt
    user_prompt = [entry['content'] for entry in eval(row['conversations']) if entry['role'] == 'user'][-1]

    # Create prompt using the extracted user content
    prompt = f"{user_prompt}\nAssistant:"

    # Generate response
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.1,
        top_k=50,
        top_p=0.1,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.pad_token_id
    )
    
    # Print results
    print(f"Row {idx + 1} User Query:\n{user_prompt}")
    print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
