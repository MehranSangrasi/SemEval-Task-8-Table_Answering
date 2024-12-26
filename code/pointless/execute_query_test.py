import pandas as pd
from queries_correction import query_gpt4
import ast
from databench_eval import Evaluator

# Initialize the evaluator
evaluator = Evaluator()

# Load the CSV files
semeval_train = pd.read_csv("data/semeval_train.csv")
wq = pd.read_csv("data/wrong_queries.csv")
table_info = pd.read_csv("data/table_info.csv")

def custom_compare(value, truth, semantic=None):
    """Custom evaluation function to compare values based on the semantic type."""
    if "list" in semantic:
        return sorted(value) == sorted(truth)
    else:
        return str(value) == str(truth)

def execute_query(dataset, question, answer, answer_type, columns_used, column_types_used, columns):
    """Execute a query on a given dataset and return the result."""
    
    df = pd.read_csv(f"data/datasets/{dataset}.csv")
    
    try:
        query = query_gpt4(dataset, question, answer, answer_type, columns_used, column_types_used, columns)
        print(query)
        
        # result = eval(query, {'df': df, 'pd': pd, 'float': float})  # Execute the query within the context of the dataset
        result = query
        row = semeval_train[semeval_train['question'] == question]
        truth = row['answer'].values[0]  # Ground truth answer
        
        print("Answer of own query:", result)
        print("Ground Truth:", truth)
        
        semantic_type = row['type'].values[0] 
        result = evaluator.compare(value=result, truth=truth, semantic=semantic_type)
        
        if result == True:
            return query
        
        tries = 0
        while result != True and tries < 5:
            print("Wrong answer, retrying...")
            query = query_gpt4(dataset, question, answer, answer_type, columns_used, column_types_used, columns)
            result = eval(query, {"df": df})
            truth = row['answer'].values[0]  # Ground truth answer
            print("Answer of own query:", result)
            print("Ground Truth:", truth)
            semantic_type = row['type'].values[0] 
            result = evaluator.compare(value=result, truth=truth, semantic=semantic_type)
            print(f'Evaluator Answer : {result}')
            tries += 1
            
            if result == True:
                return query
        
        return "FALSE"
        
        # return query
    except Exception as e:
        return f"Error: {e}"

# Process each question in the wrong queries CSV file
print("Processing questions for the datasets...")

results = []

for index, row in wq.iterrows():
    question = row['question']
    dataset_id = row['dataset']
    answer = row["answer"]
    answer_type = row["type"]
    columns_used = row["columns_used"]
    column_types_used = row["column_types"]
    
    # Extract column information for the current dataset
    columns = table_info.loc[table_info['Dataset_ID'] == dataset_id, 'Columns'].values
    columns = columns[0] if len(columns) > 0 else []  # Extract the first value or fallback to an empty list

    print(f"Processing Question {index + 1} with dataset {dataset_id}: {question}")
    
    try:
        query = execute_query(dataset_id, question, answer, answer_type, columns_used, column_types_used, columns)
        if query[0] != 'd' and query[0] != 'E':  # Adjust query formatting if needed
            query = query[1:]
        
        results.append({'question': question, 'dataset_id': dataset_id, 'columns': columns, 'query': query})
    except Exception as e:
        print(f"Error processing Question {index + 1}: {str(e)}")
        results.append({'question': question, 'dataset_id': dataset_id, 'columns': columns, 'query': f"Error: {str(e)}"})

# Save the results to a CSV file
results_df = pd.DataFrame(results)
output_file = "data/right_queries.csv"
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
