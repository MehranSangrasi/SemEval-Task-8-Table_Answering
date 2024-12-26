import pandas as pd

# Load the queries CSV file
queries = pd.read_csv("data/dev_with_queries_2.csv")

# Dictionary to cache loaded datasets
# dataset_cache = {}

# List to store the answers
answer_list = []

# Iterate over the rows of the queries DataFrame
for index, row in queries.iterrows():
    dataset = row['dataset']
    
    df = pd.read_csv(f"data/datasets/{dataset}.csv")
    
    # Check if the dataset is already loaded
    # if dataset not in dataset_cache:
    #     # Load the dataset and store it in the cache
    #     dataset_cache[dataset] = pd.read_csv(f"data/datasets/{dataset}.csv")
    
    # Get the DataFrame for the current dataset_id
    # df = dataset_cache[dataset]
    
    question = row['question']
    print(f"Processing question: {question}")
    
    query = row['queries']
    
    try:
        # Evaluate the query and append the result to the answers list
        
        answer = eval(query, {'df': df})
        print(answer)
    except Exception as e:
        # If an error occurs during evaluation, store the error message
        answer = "ERROR"
    
    answer_list.append(answer)

# Add the answers as a new column in the DataFrame
queries['own_answers'] = answer_list

# Save the updated DataFrame back to a CSV file
queries.to_csv("data/dev_with_answers.csv", index=False)

# Display the updated DataFrame
print(queries.head())
