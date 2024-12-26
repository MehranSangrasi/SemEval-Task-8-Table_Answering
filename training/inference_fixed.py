import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from datasets import load_dataset
from ast import literal_eval

model_dir="/home/mehran1/projects/def-cjhuofw-ab/mehran1/SemEval/data/models/code-llama-3-1-8b-text-to-sql"

# Load Model with PEFT adapter
model = AutoModelForCausalLM.from_pretrained(
  model_dir,
  device_map="auto",
  torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


csv_file = "data/conversations_final.csv"  # Replace with your actual file path
dataset = load_dataset("csv", data_files=csv_file)["train"]

# Extract the last two rows
test =[]



for i in range(10):
    print(i, literal_eval(dataset[i])[0])
    test.append(literal_eval(dataset[i])[0])

for i in test:
    prompt = pipe.tokenizer.apply_chat_template(i, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

    # print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
    # print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
    print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")






