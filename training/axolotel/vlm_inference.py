import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import time
import os
from vllm import SamplingParams, LLM

# Load the local Excel file
df = pd.read_csv("dev")

# Concatenate the "prompt" and "text" columns
df['combined'] = df['prompt'].astype(str) + "\n" + df['text'].astype(str)

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Use the complete dataset
prompts = dataset['combined']





# Define sampling parameters and LLM model
sampling_params = SamplingParams(max_tokens=1000)
llm = LLM(model="johnsnowlabs/JSL-MedQwen-2.5-72B-v1", tensor_parallel_size=2)

def generate_batch(prompts):
    # Ensure prompts are in the correct format
    if not isinstance(prompts, list):
        prompts = [prompts]
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

# Check if the output file exists and determine the starting index
output_file = "generated_responses.xlsx"
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    start_index = existing_df['response'].last_valid_index() + 1
    generated_text = existing_df['response'].tolist()
else:
    start_index = 0
    generated_text = []

# Split prompts into batches starting from the determined index
batch_size = 8  # Adjust batch size as needed
batches = [prompts[i:i + batch_size] for i in range(start_index, len(prompts), batch_size)]

time_taken = 0

# Generate text in batches
for batch_index, batch in enumerate(tqdm(batches, initial=start_index // batch_size, total=len(prompts) // batch_size)):
    start = time.time()
    batch_results = generate_batch(batch)
    taken = time.time() - start
    time_taken += taken
    
    # Print responses after each batch
    for response in batch_results:
        print(response)
    
    generated_text.extend(batch_results)
    
    # Save responses to an Excel file after each batch
    response_df = pd.DataFrame({'prompt': prompts[:len(generated_text)], 'response': generated_text})
    response_df.to_excel(output_file, index=False)

# Count tokens in generated text
tokens = sum(len(sample.split()) for sample in generated_text)

print(tokens)
print("tok/s", tokens // time_taken)