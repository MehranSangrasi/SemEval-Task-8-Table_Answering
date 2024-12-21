import pandas as pd
from databench_eval import Evaluator


df = pd.read_csv("data/pointless/dev_with_answers.csv")
evaluator = Evaluator()

correct=0
for index, row in df.iterrows():
    
    actual_answer = row["answer"]
    print("Actual answer: ", actual_answer)
    predicted_answer = row["own_answers"]
    print("Predicted answer: ", predicted_answer)
    print("\n\n")
    
    semantic = row["type"]
    
    result = evaluator.compare(value=predicted_answer, truth=actual_answer, semantic=semantic)
    
    if result == True:
        correct+=1
        


print("Accuracy: ", correct/len(df))