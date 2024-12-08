from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch

# Load the dataset
print("Loading dataset from CSV...")
file_path = "table_info.csv"
df = pd.read_csv(file_path)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("arcee-ai/Llama-3.1-SuperNova-Lite")
model = AutoModelForCausalLM.from_pretrained("arcee-ai/Llama-3.1-SuperNova-Lite", torch_dtype=torch.float16)

# Set pad_token_id explicitly
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# Create pipeline with GPU support
llama = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # Use device=-1 for CPU

# Define function to query the model
def query_model(dataset_id, question):
    # Filter dataset by Dataset_ID
    filtered_df = df[df['Dataset_ID'] == dataset_id]

    if filtered_df.empty:
        return "Dataset not found."

    # Truncate columns to reduce size
    columns = filtered_df['Columns'].values[0][:500]

    # Prepare prompt
    prompt = f"""
    Dataset summary:
    {columns}

    Question: {question}

    Answer:
    """
    try:
        response = llama(prompt, max_new_tokens=200)
        return response[0]['generated_text']
    except Exception as e:
        return f"Error during generation: {e}"

# Example query
print("Asking questions about the dataset...")
question = "Could you list the lower 3 fare ranges by number of survivors?"
dataset_id = "002_Titanic"
answer = query_model(dataset_id, question)
print(f"Answer: {answer}")
