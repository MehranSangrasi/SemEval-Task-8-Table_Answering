import pandas as pd
from databench_eval import Evaluator
import ast
import json
import re


# Initialize the evaluator
evaluator = Evaluator()

# Load the CSV files
# dev_set = pd.read_csv("data/dev_answers.csv")
dev = json.load(open("data/dev.json"))
# dev_set = pd.DataFrame(dev)
dev_set = json.load(open("data/dev_output.json"))
dev_csv = pd.read_csv("data/dev_set.csv")
correct = 0
count = 0

for index, items in enumerate(dev_set):
    try:
        # question = row["question"]
        question = dev[index]["input"]
        question = question.split("\n")[0]
        question = question.replace("Question: ", "")
        question = question.rstrip()
        # print(question)
        dataset = dev_csv[dev_csv["question"] == question]["dataset"].values
        dataset = dataset[0]
        # print(dataset)
        
        df = pd.read_csv(f"data/datasets/{dataset}.csv")
        
        text = items["output"]
        
        start_index = text.find('df')
        end_index = text.find('\n', start_index)

        if start_index != -1 and end_index != -1:
            extracted_text = text[start_index:end_index]
            answer = eval(extracted_text)
            # print("Extracted text:", extracted_text)
        else:
            if end_index == -1:
                extracted_text = text[start_index:]
                print(extracted_text, end="\n\n")
                answer = eval(extracted_text)

        
        actual_answer = dev_csv[dev_csv["question"] == question]["answer"].values[0]
        semantic = dev_csv[dev_csv["question"] == question]["type"].values[0]
        
        result = evaluator.compare(value=answer, truth=actual_answer, semantic=semantic)
        
        if result == True:
            correct+=1
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # df = pd.read_csv(f"data/datasets/{dataset}.csv")
        
        # query = row['queries'].splitlines()[0]
        # predicted_answer = eval(query)
        # print("Question:", row['question'])
        # print("Our answer:", predicted_answer)
        # print("Actual answer:", answer)
        
        # semantic_type = row['type']
        # result = evaluator.compare(value=predicted_answer, truth=answer, semantic=semantic_type)
        # print("Equal or not:", result, end="\n\n")
        
        # if result == True:
        #     correct += 1
    
    except Exception as e:
        # print(query)
        print(e)
        continue
    
# print(count)
    
print(len(dev_set))
print(correct)
accuracy = correct/len(dev_set)
print(accuracy)