import pandas as pd
from databench_eval import Evaluator
from datasets import load_dataset
import ast


# Initialize the evaluator
evaluator = Evaluator()

# Load the CSV files
dev_set = pd.read_csv("data/dev_set_queries.csv")
correct = 0
error = 0
wrong_queries = []
error_queries = []
correct_queries = []
ind=0
# new_dev = pd.read_csv("data/train_set.csv")

with open("data/databench_lite.txt", "w") as f: 
    for index, row in dev_set.iterrows():
        try:
            ind+=1
            question = row["question"]
            print("Question:", row['question'])
            dataset = row["dataset"]
            # answer = new_dev[new_dev['question'] == question]["answer"].values
            answer = row['sample_answer']
            
            df = pd.read_csv(f"data/datasets/{dataset}_sample.csv")
            
            query = row['queries']
            print("Query:", query)
            predicted_answer = eval(query, {"df": df, "pd": pd})
            
            print("Our answer:", predicted_answer)
            print("Actual answer:", answer)
            
            
            f.write(str(predicted_answer) + "\n")
            
            semantic_type = row['type']
            result = evaluator.compare(value=predicted_answer, truth=answer, semantic=semantic_type)
            print("Equal or not:", result, end="\n\n")
            
            if result == True:
                correct += 1
                # correct_queries.append(row.to_dict())
                
            # else:
            #     wrong_queries.append(row.to_dict())
        
        except Exception as e:
            print("Error query: ", row['queries'])
            # error+=1
            # error_queries.append(row.to_dict())
            f.write(str(row['queries'].strip("\n")) + "\n")
            
            print(e, end="\n\n\n")
            continue
        
# print("Error: ", error)
print(ind)
print("Correct: ", correct)
# print("Giving wrong answer: ", len(wrong_queries))
accuracy = correct/len(dev_set)
print(accuracy)


# correct_queries_df = pd.DataFrame(correct_queries)
# correct_queries_df.to_csv("data/correct_queries_from_wrong.csv", index=False)

# error_queries_df = pd.DataFrame(error_queries)
# error_queries_df.to_csv("data/error_queries_from_wrong.csv", index=False)

# wrong_queries_df = pd.DataFrame(wrong_queries)
# wrong_queries_df.to_csv("data/wrong_queries_qwen_from_wrong.csv", index=False)