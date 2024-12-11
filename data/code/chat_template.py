import pandas as pd


data = pd.read_csv("data/matched_data_2.csv")

new_data = []

def map_template(prompt, answer):
    
    return [ {"from": "human", "value": f"{prompt}" }, 
            { "from": "gpt", "value": f"{answer}" }
            ]
    
    
for index, row in data.iterrows():
    
    prompt = f"Question: {row['question']} \n Dataset Columns: {row['columns']}"
    
    answer = row['query']
    
    conversation = map_template(prompt, answer)
    print(conversation)
    new_data.append({"conversations": conversation})
    

conversations_df = pd.DataFrame(new_data)

conversations_df.to_csv("data/conversations.csv", index=False)
    
    
    
    
    