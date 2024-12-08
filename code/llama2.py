# Import required libraries
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Step 1: Load the pre-trained Llama 2 model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with the correct model name if using another variant
print("Loading pre-trained Llama 2 model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Step 2: Load the Titanic dataset
print("Loading Titanic dataset from CSV...")
file_path = "train.csv"  # Replace with the correct path to your Titanic dataset
df = pd.read_csv(file_path)

# Step 3: Prepare the dataset as a text summary
print("Preparing dataset for Llama 2...")
# Example: Create a summary of key statistics (customize as needed)
data_summary = df.describe(include="all").to_string()

# Convert the dataset into a textual format
dataset_text = f"""
Here is the summary of the Titanic dataset:
{data_summary}

The dataset has {len(df)} rows and the following columns:
{', '.join(df.columns.tolist())}
"""

# Step 4: Set up the Llama 2 pipeline
print("Setting up Llama 2 pipeline...")
llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Step 5: Define a function to query the model
def query_llama(dataset_text, question):
    prompt = f"""
You are an expert data analyst. I will provide a dataset summary, and you will answer questions based on it.

Dataset summary:
{dataset_text}

Question: {question}

Answer:
"""
    response = llama_pipeline(prompt, max_length=512, temperature=0.7, top_p=0.9)
    return response[0]["generated_text"]

# Step 6: Ask questions about the dataset
print("Asking questions about the Titanic dataset...")

# Example Question 1: Maximum age of passengers
question1 = "What is the maximum age of the passengers?"
answer1 = query_llama(dataset_text, question1)
print(f"Question: {question1}")
print(f"Answer: {answer1}")

# Example Question 2: Average age of passengers
question2 = "What is the average age of the passengers?"
answer2 = query_llama(dataset_text, question2)
print(f"Question: {question2}")
print(f"Answer: {answer2}")

# Example Question 3: Total number of survivors
question3 = "How many passengers survived?"
answer3 = query_llama(dataset_text, question3)
print(f"Question: {question3}")
print(f"Answer: {answer3}")
