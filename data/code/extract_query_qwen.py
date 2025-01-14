import pandas as pd
# from databench_eval import Evaluator
import ast
from datasets import load_dataset


# Initialize the evaluator
# evaluator = Evaluator()

# dev = load_dataset("cardiffnlp/databench", name="semeval", split="dev")
# dev_set_new = pd.DataFrame(dev)
# dev_set_new.to_csv("data/dev_set_2.csv", index=False)

# Load the CSV files
dev_set = pd.read_csv("competition/competition/test_qa.csv")
correct = 0
# new_dev = pd.read_csv("data/train_set.csv")
query_df = pd.read_csv("data/test_with_queries_qwen_unsloth_2.csv")
queries = []


for index, row in query_df.iterrows():
    try:
        query_string = row['queries']
        start_marker = "assistant\n"
        end_marker = "\nuser"

        # Extract the query
        start_index = query_string.find(start_marker) + len(start_marker)
        end_index = query_string.find(end_marker, start_index)

        extracted_query = query_string[start_index:end_index].strip()
        
        queries.append(extracted_query)
        
        print(extracted_query)
                
        
        
    
    except Exception as e:
        print("Error query: ", row['query'])
        # queries.append(row['query'])
        print(e, end="\n\n\n")
        continue
    

dev_set['queries'] = queries

dev_set.to_csv("data/test_qa_queries.csv", index=False)
    
# accuracy = correct/total
# print(accuracy)
