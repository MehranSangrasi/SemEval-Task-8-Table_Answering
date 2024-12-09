import pandas as pd
from datasets import load_dataset
# import numpy as np
# Load the dataset
# df = 
# df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{dataset}/all.parquet")
# print(df.info())

queries = pd.read_csv("data/queries_final.csv")

answer_list = []

for index, row in queries.iterrows():
    dataset = row['dataset_id']
    df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{dataset}/all.parquet")
    
    question = row['question']
    print(question)
    
    
    if isinstance(row['answer'], str):
        query = row['answer'].strip('"')
    else:
        query = row['answer']
        
        
    
    try:
        # Evaluate the query and append the result to the answers list
        answer = eval(query, {'df': df})
        print(answer)
    except Exception as e:
        # If an error occurs during evaluation, store the error message
        answer = "ERROR"
    
    answer_list.append(answer)
    
    
    
    
    
queries['answers'] = answer_list
queries.to_csv("data/queries_with_answers.csv", index=False)


