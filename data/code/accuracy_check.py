import pandas as pd
from databench_eval import Evaluator
import ast


# Initialize the evaluator
evaluator = Evaluator()

# Load the CSV files
dev_set = pd.read_csv("data/dev_answers.csv")
correct = 0


for index, row in dev_set.iterrows():
    try:
        # question = row["question"]
        dataset = row["dataset"]
        answer = row["sample_answer"]
        # answer_type = row["type"]
        
        df = pd.read_csv(f"data/datasets/{dataset}.csv")
        
        query = row['queries'].splitlines()[0]
        predicted_answer = eval(query)
        print("Question:", row['question'])
        print("Our answer:", predicted_answer)
        print("Actual answer:", answer)
        
        semantic_type = row['type']
        result = evaluator.compare(value=predicted_answer, truth=answer, semantic=semantic_type)
        print("Equal or not:", result, end="\n\n")
        
        if result == True:
            correct += 1
    
    except Exception as e:
        print(query)
        print(e)
        continue
    
print(len(dev_set))
accuracy = correct/len(dev_set)
print(accuracy)