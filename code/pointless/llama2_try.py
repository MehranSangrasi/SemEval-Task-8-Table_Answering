from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch

# Load the dataset
print("Loading dataset from CSV...")
file_path = "table_info.csv"  # Ensure correct path
df = pd.read_csv(file_path)




tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# Use the pipeline to interact with the model
llama = pipeline("text-generation", model=model, tokenizer=tokenizer ,device=0,     torch_dtype=torch.float16  # Reduces memory usage
)

# Define function to query the new model
def query_model(dataset_id, question):
    # Filter dataset by Dataset_ID
    filtered_df = df[df['Dataset_ID'] == dataset_id]

    if filtered_df.empty:
        return "Dataset not found."

    # Extract only the relevant columns for this question to reduce dataset size
    columns = filtered_df['Columns'].values[0]
    # Limit the columns shown to avoid excess output length (e.g., first 10 columns or based on question relevance)
    # limited_columns = ', '.join(columns.split(',')[:10])  # limit to first 10 columns if too long

    # Prepare the prompt for the new model
    prompt = f"""
I will provide a dataset summary, and you need to generate a python code based on the question so I can run it and find the answer. Provide nothing else but the dataframe query python code only.
Dont give explanation. You can provide multiple lines of query code if needed. DONT EXPLAIN. JUST GIVE THE QUERY.
For example, "Among those who survived, which fare range was the most common: (0-50, 50-100, 100-150, 150+)?", you should provide the following code:
```
fare_bins = [0, 50, 100, 150, float('inf')]
fare_labels = ['0-50', '50-100', '100-150', '150+']
df['Fare Range'] = pd.cut(df['Fare'], bins=fare_bins, labels=fare_labels)
df.groupby('Fare Range')['Survived'].sum().nsmallest(3).index.tolist()

```
or another example,  "Does the youngest billionaire identify as male?":
```
df.loc[df['age'].notnull()].sort_values('age').iloc[0]['gender'] == 'Male' 
```
Example of output types:
how many: 3(int)
which of these is(are) or has(have) : [1,2,3], [steve, bill, elon] (list) or 3.14 (single) note: observe the singularity or multiple value requirement
is there: True/False (bool)
what is: 3.14(single value)
what are: [1,2,3] (list)

Dataset form multiple columns of: (column_name);(data_type)
for_example:  rank; uint16, personName; category, age; float64, finalWorth; uint32

Dataset summary:
{columns}

Question: {question}


Answer:
"""
    # Send query to the model
    response = llama(prompt, max_new_tokens=1000)  # Adjust max_length as needed

    # Return the answer from the model's response
    return response[0]['generated_text']

# Example query usage
print("Asking questions about the dataset...")

question_2 = "Could you list the lower 3 fare ranges by number of survivors: (0-50, 50-100, 100-150, 150+)?"
dataset_2 = "002_Titanic"
answer2 = query_model(dataset_2, question_2)

print(f"Answer: {answer2}")
