# import pandas as pd
# import ast

# test_set = pd.read_csv("data/test_qa_queries.csv")



# for index, row in test_set.iterrows():
    
#     query = row['queries']
#     dataset = row['dataset']
    
#     df = pd.read_csv(f"data/test_set/{dataset}.csv")
#     # df_lite = pd.read_csv(f"data/test_set/{dataset}_sample.csv")
    
#     answer = eval(query, {'df': df, 'pd': pd})
#     # answer_lite = eval(query, {'df': df_lite, 'pd': pd})
    
    
# for index, row in test_set.iterrows():
    
#     query = row['queries']
#     dataset = row['dataset']
    
#     # df = pd.read_csv(f"data/test_set/{dataset}.csv")
#     df_lite = pd.read_csv(f"data/test_set/{dataset}_sample.csv")
    
#     # answer = eval(query, {'df': df, 'pd': pd})
#     answer_lite = eval(query, {'df': df_lite, 'pd': pd})
    
    
import pandas as pd

# Read the test queries dataset
test_set = pd.read_csv("data/test_qa_queries.csv")

# Open the files to save the answers
with open("data/predictions.txt", "w") as predictions_file, open("data/predictions_lite.txt", "w") as predictions_lite_file:
    
    # First loop: Process the full dataset
    
    print("**************************DATABENCH***************************************")
    for index, row in test_set.iterrows():
        query = row['queries']
        dataset = row['dataset']
        question = row['question']
        
        # Load the full dataset
        df = pd.read_csv(f"data/test_set/{dataset}.csv")
        
        # Evaluate the query
        try:
            answer = eval(query, {'df': df, 'pd': pd})
            predictions_file.write(f"{answer}\n")
        except Exception as e:
            print("Index: ", index+2)
            print("Question: ", question)
            print("Query: ", query)
            print("Error: ", e, end="\n\n")
            predictions_file.write(f"Error: {query}\n")
            
    print("**************************DATABENCH LITE***************************************")
    
    # Second loop: Process the sampled dataset
    for index, row in test_set.iterrows():
        query = row['queries']
        dataset = row['dataset']
        
        # Load the sampled dataset
        df_lite = pd.read_csv(f"data/test_set/{dataset}_sample.csv")
        
        # Evaluate the query
        try:
            answer_lite = eval(query, {'df': df_lite, 'pd': pd})
            # answer_lite = answer_lite.strip("\n")
            predictions_lite_file.write(f"{answer_lite}\n")
        except Exception as e:
            print("Index: ", index+2)
            print("Question: ", question)
            print("Query: ", query)
            print("Error: ", e, end="\n\n")
            predictions_file.write(f"Error: {query}\n")

    
    
    
    
    