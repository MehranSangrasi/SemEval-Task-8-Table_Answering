import json
import pandas as pd

with open('data/training.json') as f:
    data = json.load(f)
    

instruct = {"text":[]}


for item in data["messages"]:
    
    instruction = item["instruction"].strip("\n")
    
    message = f"<s>[INST] <<SYS>> {instruction} <</SYS>> {item['input']} [/INST] {item['output']} </s>"
    print(message)
    instruct["text"].append(message)
    
df = pd.DataFrame(instruct)
df.to_csv('data/training_test.csv', index=False)
print("CSV file created successfully!")