import pandas as pd
from databench_eval import Evaluator


cw = pd.read_csv("data/correct_from_wrong.csv")
wq = pd.read_csv("data/wrong_queries.csv")
evaluator = Evaluator()

correct=0
for index, row in cw.iterrows():
    
    dataset = row["dataset_id"]
    
    df = pd.read_csv(f"data/datasets/{dataset}.csv")
    
    query = row["query"]
    
    pred_answer = eval(query, {"df": df, "pd": pd})
    # print(answer)
    
    question = row["question"]
    
    true_answer = wq.loc[wq["question"] == question]["answer"].values[0]
    # print(true_answer)
    
    semantic = wq.loc[wq["question"] == question]["type"].values[0]
    
    # print(semantic)

    
    # semantic = row["type"]
    
    result = evaluator.compare(value=pred_answer, truth=true_answer, semantic=semantic)
    
    if result == True:
        correct+=1
        


print("Accuracy: ", correct/len(cw))