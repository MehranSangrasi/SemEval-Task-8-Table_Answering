import pandas as pd
import json


with open("data/dev.json", "r") as f:
    eval = json.load(f)
    
conversations = []

system_message = """
You are Qwen, a helpful assistant designed for coding. You are supposed to generate single line python pandas data-frame executable query for tabular Q/A given a question and its dataset columns with some columns having unique values.
"""
    
    
for item in eval:
    user_message = item["input"]
        

    # Split the string into the question and the dataset columns
    parts = user_message.split("Dataset Columns:")
    question = parts[0]
    columns = parts[1].strip()

    # Use regex to split only on commas that are followed by a space and a word (indicating a new column)
    import re
    formatted_columns = re.sub(r', (?=\w+ \()', '\n', columns)

    # Combine the question and formatted columns
    input = f"{question}Dataset Columns:\n{formatted_columns}"
    
    input_string = system_message + "\n\n" + input
    
    conversations.append([{"from": "human", "value": input_string}])


conversations = pd.DataFrame(conversations)
conversations.to_csv("data/eval_convo_2.csv", index=False)
    
    
    
    