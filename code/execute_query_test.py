import pandas as pd
# from data.code.dev_set_accuracy import convert_to_value
from queries_correction import query_gpt4
import ast
from databench_eval import Evaluator

evaluator = Evaluator()


# Load the CSV file
semeval_train = pd.read_csv("data/semeval_train.csv")
df = pd.read_csv("data/wrong_queries.csv")
table_info = pd.read_csv("data/table_info.csv")
# error = pd.read_csv("data/error_rows.")

def custom_compare(value, truth, semantic=None):
    """ Custom evaluation function. """
    import pdb; pdb.set_trace()
    if "list" in semantic:
        return sorted(value) == sorted(truth)
    else:
        return str(value) == str(truth)

def convert_to_value(val):
    """
    Convert a string to a Python value (boolean, list, int, or float).
    """
    # Convert boolean strings to actual booleans
    # import pdb; pdb.set_trace()
    try:
        if val.lower() == 'true':
            return True
        elif val.lower() == 'false':
            return False
    except:
        pass

    try:# If the value starts and ends with brackets, try to convert it to a list
        if val.startswith("[") and val.endswith("]"):
            val = ast.literal_eval(val)
            val = sorted(val)
            return val
    except:
        pass
    # Try to convert to float or int
    try:
        if val.replace(".", "", 1).isdigit() or val[0] == '-' and val[1:].replace(".", "", 1).isdigit():
            # Check for float or integer
            try:
                if "." in val:
                    val = float(val)
                    val = round(val, 2)
                    # import pdb; pdb.set_trace()
                else:
                    return int(val)
            except ValueError:
                pass  # If the value can't be converted, return the original string
    except:
        pass

    return val  # Return the original value if it doesn't match any of the above cases



# def execute_query(dataset, query, question):

def execute_query(dataset, question, answer, answer_type, columns_used, column_types_used, columns):
    """
    Execute a query on a given dataset and return the result.
    """
    
    # correct = 0
    # Load the dataset
    df = pd.read_csv(f"data/datasets/{dataset}.csv")
    
    try:
        # Evaluate the query and return the result
        query = query_gpt4(dataset_id, question, answer, answer_type,columns_used, column_types_used, columns)
        # import pdb; pdb.set_trace()
        print(query)
        
        result = eval(query, {'df': df})
        row = semeval_train[semeval_train['question'] == question]
        
        truth = row['answer'].values[0]  # Ground truth answer
        
        print(result)
        print(truth)
        
        semantic_type = row['type'].values[0] 
        
        result = evaluator.compare(value=result, truth=truth, semantic=semantic_type)
        print(f'Evaluator Answer : {result}')
        
        # result = convert_to_value(result)
        # print(result)
        # answer = convert_to_value(row['answer'].values[0])
        # print(answer)
        
        tries = 0
        while result != True and tries<5:
            print("wrong")
            
            query = query_gpt4(dataset, question, answer,answer_type, columns_used, column_types_used, columns)
            
            result = eval(query, {"df": df})
            
            row = semeval_train[semeval_train['question'] == question]

            # import pdb; pdb.set_trace()
            truth = row['answer'].values[0]  # Ground truth answer
            
            print(result)
            print(truth)
            semantic_type = row['type'].values[0] 
            result = evaluator.compare(value=result, truth=truth, semantic=semantic_type)
            print(f'Evaluator Answer : {result}')
            
            # result = convert_to_value(result)
            # print(result)
            tries+=1
            
        
        # print("right")
        return query
    except Exception as e:
        # If an error occurs during evaluation, return an error message
        return f"Error: {e}"
    
    
# Iterate through questions and datasets
print("Asking questions about the dataset...")

results = []

for index, row in df.iterrows():
    question = row['question']
    dataset_id = row['dataset']
    answer = row["answer"]
    answer_type = row["type"]
    columns_used = row["columns_used"]
    column_types_used = row["column_types"]
    
    # Extract columns properly
    columns = table_info.loc[table_info['Dataset_ID'] == dataset_id, 'Columns'].values
    if len(columns) > 0:
        columns = columns[0]  # Extract the first value from the list
    else:
        columns = []  # Fallback to an empty list if no columns are found

    print(f"Processing Question {index + 1} with dataset {dataset_id}: {question}")
    
    try:
        # Execute the query and store the result
        query = execute_query(dataset_id, question, answer, answer_type, columns_used, column_types_used, columns)
        # print(query)
        # print(query)
        if query[0] != 'd' and query[0] != 'E':
            query = query[1:]
        # import pdb; pdb.set_trace()
        
        
        results.append({'question': question, 'dataset_id': dataset_id, 'columns': columns, 'query': query})

    except Exception as e:
        print(f"Error processing Question {index + 1}: {str(e)}")
        results.append({'question': question, 'dataset_id': dataset_id, 'columns': columns, 'query': f"Error: {str(e)}"})

# Save results to a CSV file
results_df = pd.DataFrame(results)
output_file = "data/correct_queries.csv"
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
