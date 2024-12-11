from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
model_dir = "/home/mehran1/projects/def-cjhuofw-ab/mehran1/SemEval/training/model_200"
tokenizer_dir = "/home/mehran1/projects/def-cjhuofw-ab/mehran1/SemEval/training/tokenizer_200"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

# Set model to evaluation mode
model.eval()

# Preprocessing function for inference
def preprocess_inference(question, dataset_columns):
    input_text = f"Question: {question} \n Dataset Columns: {', '.join(dataset_columns)}"
    tokenized_inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    return tokenized_inputs

# Inference example
question = "Is the person with the highest net worth self-made?"
inputs = preprocess_inference(question, ["rank", "personName", "age"])
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate response
output = model.generate(
    input_ids=inputs["input_ids"], max_new_tokens=128
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated Response:", generated_text)
