import json


with open('train_dev_test/axolotl/transformed/alpaca_trans.json') as f:
    data = json.load(f)
    
json_obj = {"conversations": []}
    
for item in data:
    conversation = []
    
    conversation_human = {"from": "human", "value": item['instruction']}
    conversation_gpt = {"from": "assistant", "value": item['output']}
    
    conversation.append(conversation_human)
    conversation.append(conversation_gpt)
    
    json_obj["conversations"].append(conversation)
    
print(len(json_obj))

with open('train_dev_test/axolotl/transformed/chatml_trans_3.json', 'w') as f:
    json.dump(json_obj, f, indent=4)
