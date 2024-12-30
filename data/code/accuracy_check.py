import pandas as pd
from databench_eval import Evaluator
import ast


# Initialize the evaluator
evaluator = Evaluator()

# Load the CSV files
dev_set = pd.read_csv("data/wrong_queries.csv")
correct = 0
error = 0
wrong_queries = []
error_queries = []
correct_queries = []
new_dev = pd.read_csv("data/train_set.csv")


for index, row in dev_set.head(100).iterrows():
    try:
        question = row["question"]
        print("Question:", row['question'])
        dataset = row["dataset"]
        answer = new_dev[new_dev['question'] == question]["answer"].values
        answer = answer[0]
        
        df = pd.read_csv(f"data/datasets2/{dataset}.csv")
        
        query = row['query']
        print("Query:", query)
        predicted_answer = eval(query, {"df": df, "pd": pd})
        
        print("Our answer:", predicted_answer)
        print("Actual answer:", answer)
        
        semantic_type = row['type']
        result = evaluator.compare(value=predicted_answer, truth=answer, semantic=semantic_type)
        print("Equal or not:", result, end="\n\n")
        
        if result == True:
            correct += 1
            correct_queries.append(row.to_dict())
            
        else:
            wrong_queries.append(row.to_dict())
    
    except Exception as e:
        print("Error query: ", row['query'])
        error+=1
        error_queries.append(row.to_dict())
        print(e, end="\n\n\n")
        continue
    
print("Error: ", error)
print("Correct: ", correct)
print("Giving wrong answer: ", len(wrong_queries))
accuracy = correct/100
print(accuracy)


correct_queries_df = pd.DataFrame(correct_queries)
correct_queries_df.to_csv("data/correct_queries_from_wrong.csv", index=False)

error_queries_df = pd.DataFrame(error_queries)
error_queries_df.to_csv("data/error_queries_from_wrong.csv", index=False)

wrong_queries_df = pd.DataFrame(wrong_queries)
wrong_queries_df.to_csv("data/wrong_queries_qwen_from_wrong.csv", index=False)