import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Load the fine-tuned model and tokenizer
model_dir = "/Users/mehran/Downloads/model_200"
tokenizer_dir = "/Users/mehran/Downloads/tokenizer_200"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name_or_path=model_dir,
    tokenizer_name_or_path=tokenizer_dir,
    load_in_4bit=True,
    dtype=None,
)

# Ensure the model is set for inference
model = FastLanguageModel.for_inference(model)
model.eval()  # Set the model to evaluation mode
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Preprocessing function for dataset

def preprocess_inference(question, dataset_columns):
    """Preprocess the input question and dataset columns for inference."""
    input_text = f"Question: {question} \n Dataset Columns: {', '.join(dataset_columns)}"
    tokenized_inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    return tokenized_inputs

# Load and preprocess the dataset
csv_file = "/Users/mehran/CodeSpaces/Testing/Table_Answering/data/matched_data_2.csv"
dataset = load_dataset("csv", data_files=csv_file)["train"]

columns = "rank; uint16, personName; category, age; float64, finalWorth; uint32, category; category, source; category, country; category, state; category, city; category, organization; category, selfMade; bool, gender; category, birthDate; datetime64[us, UTC], title; category, philanthropyScore; float64, bio; object, about; object"
dataset_columns = [col.split(";")[0].strip() for col in columns.split(",")]

# Calculate loss on the dataset
def calculate_loss(model, dataset):
    """Calculate the average loss over the dataset."""
    total_loss = 0
    num_samples = 0

    for example in dataset:
        question = example["question"]
        target = example["query"]

        # Preprocess inputs and targets
        inputs = preprocess_inference(question, dataset_columns)
        labels = tokenizer(
            target, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = labels["input_ids"].to(device)

        # Compute model outputs and loss
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
            num_samples += 1

    return total_loss / num_samples

# Example usage
data_loss = calculate_loss(model, dataset)
print(f"Average Loss on Dataset: {data_loss}")

# Perform inference
question = "Is the person with the highest net worth self-made?"
inputs = preprocess_inference(question, dataset_columns)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Set up the TextStreamer for real-time output
text_streamer = TextStreamer(tokenizer)

# Generate a response
output = model.generate(
    input_ids=inputs["input_ids"],
    max_new_tokens=128,
    streamer=text_streamer,
    use_cache=True,
)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated Response:", generated_text)
