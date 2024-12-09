from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch

# Load the dataset
print("Loading dataset from CSV...")
file_path = "table_info.csv"  # Ensure the file path is correct
df = pd.read_csv(file_path)

# Load tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("arcee-ai/Llama-3.1-SuperNova-Lite")
model = AutoModelForCausalLM.from_pretrained(
    "arcee-ai/Llama-3.1-SuperNova-Lite",
    torch_dtype=torch.float16,  # Reduce memory usage with mixed precision
    device_map="auto"  # Ensure the model uses MPS if available
)

# Check and set `pad_token_id`
if model.config.pad_token_id is None:
    print("Setting pad_token_id...")
    model.config.pad_token_id = tokenizer.pad_token_id

# Initialize the pipeline
print("Initializing pipeline...")
llama = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=0,  # Use MPS (0 refers to GPU index; ensure compatibility)
    pad_token_id=tokenizer.pad_token_id  # Ensure padding token is correctly set
)

# Define the function to query the model
def query_model(dataset_id, question):
    # Filter dataset by Dataset_ID
    filtered_df = df[df['Dataset_ID'] == dataset_id]

    if filtered_df.empty:
        return "Dataset not found."

    # Prepare a minimal prompt for debugging purposes
    prompt = "Hello"

    try:
        # Generate response
        print("Generating response...")
        response = llama(prompt, max_new_tokens=50)  # Limit token generation to avoid memory issues
        return response[0]['generated_text']
    except Exception as e:
        return f"Error during generation: {e}"

# Example query usage
print("Asking questions about the dataset...")
question = "Hello"
dataset_id = "002_Titanic"
answer = query_model(dataset_id, question)
print(f"Answer: {answer}")
