import pandas as pd


wq = pd.read_csv("data/wrong_queries.csv")

fw = pd.read_csv("data/final_wrong.csv")

wrong_questions = []


for index, row in fw.iterrows():
    
    question = row["question"]
    
    wrong_row = wq.loc[wq["question"] == question]
    
    wrong_questions.append(wrong_row)
    

wrong_questions_df = pd.concat(wrong_questions, ignore_index=True)
wrong_questions_df.to_csv("data/wrong_questions.csv", index=False)
    
    
    
    