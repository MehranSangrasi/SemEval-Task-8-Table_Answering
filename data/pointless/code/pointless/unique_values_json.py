import json

messages = {"messages": []}

with open("data/dev.json", "r") as f:
    train_data = json.load(f)


for item in train_data:
    input = item["input"]

    messages["messages"].append(
        [
            {
                "role": "system",
                "content": "You are Qwen, a helpful assistant designed for coding. You are supposed to generate single line python pandas data-frame executable query for tabular Q/A given a question and its dataset columns with some columns having unique values.",
            },
            {"role": "user", "content": input}
        ]
    )
    
print(len)
with open("data/eval.json", "w") as f:
    json.dump(messages, f)
    

