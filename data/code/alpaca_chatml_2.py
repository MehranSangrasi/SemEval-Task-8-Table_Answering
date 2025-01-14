import json


with open('train_dev_test/axolotl/transformed/alpaca_trans_2.json') as f:
    data = json.load(f)
    
json_obj = []
    
conversations = {"conversations":[]}
for item in data:
    conversations = {"conversations":[]}
    
    conversation_human = {"from": "human", "value": item['instruction']}
    conversation_gpt = {"from": "assistant", "value": item['output']}
    
    conversations["conversations"].append(conversation_human)
    conversations["conversations"].append(conversation_gpt)
    
    json_obj.append(conversations)
    
print(len(json_obj))

with open('train_dev_test/axolotl/transformed/chatml_trans_2.json', 'w') as f:
    json.dump(json_obj, f, indent=4)
