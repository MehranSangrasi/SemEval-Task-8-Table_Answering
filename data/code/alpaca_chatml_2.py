import json
import pandas as pd


# with open('data/dev_set.csv') as f:
#     dev = json.load(f)

dev_set = pd.read_csv("data/dev_set.csv")
columns = pd.read_csv("data/updated_columns_with_unique_values.csv")
    
json_obj = {"conversations": []}


for index, row in dev_set.iterrows():
    
    conversation = []
    
    question = row['question']
    dataset_col = columns[columns['dataset_id'] == row['dataset']]['columns']
    
    human_prompt = f"Question: {question} \n Dataset Columns:\n{dataset_col}"
    
    conversation_human = {"from": "human", "value": human_prompt}
    conversation_gpt = {"from": "assistant", "value": ""}
    
    conversation.append(conversation_human)
    conversation.append(conversation_gpt)
    
    json_obj["conversations"].append(conversation)
    
# for item in data:
#     conversation = []
    
#     conversation_human = {"from": "human", "value": item['instruction']}
#     # conversation_gpt = {"from": "assistant", "value": item['output']}
#     conversation_gpt = {"from": "gpt", "value": item['output']}
    
#     conversation.append(conversation_human)
#     conversation.append(conversation_gpt)
    
#     json_obj["conversations"].append(conversation)
    
print(len(json_obj["conversations"]))

with open('train_dev_test/axolotl/transformed/chatml_dev.json', 'w') as f:
    json.dump(json_obj, f, indent=4)

# dataframe = pd.DataFrame(json_obj)

# dataframe.to_csv("train_dev_test/unsloth/transformed/chatml_trans.csv", index=False)
