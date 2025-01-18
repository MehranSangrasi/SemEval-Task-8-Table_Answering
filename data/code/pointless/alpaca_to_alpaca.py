import json
import random

with open('data/train.json') as f:
    data = json.load(f)
    
with open("data/updated_queries.json") as f:
    other_data = json.load(f)

# with open('data/chatml_69.json') as f:
#     current_data = json.load(f)
    
with open("data/updated_queries_transformation.json") as f:
    final_data = json.load(f)
    
print(len(data))
print(len(other_data['messages']))
print(len(final_data))

json_obj = []

for item in data:
    # conversations = []
    input = item['input']
    output = item['output']
    # human_prompt = system_prompt + "\n\n\n" + input
    assistant_prompt = output
    
    start_index = input.find("Dataset Columns: ")
    question = input[:start_index]
    # print(question)
    start_index = start_index + len("Dataset Columns: ")
    
    dataset_description = input[start_index:]
    # print(dataset_description)
    columns = dataset_description.split('), ')
    formatted_columns = [col + ')' if not col.endswith(')') else col for col in columns]

    # Join the columns with a newline
    formatted_description = '\n'.join(formatted_columns)

    # Print the formatted description
    # print(formatted_description)
    # print("\n\n")
    
    human_prompt = question.rstrip()+" \n "+"Dataset Columns:\n"+formatted_description
    
    
    conversations = {"instruction": human_prompt, "input": "", "output": assistant_prompt}
    json_obj.append(conversations)
    
for item in other_data['messages']:

    original = item['output']
    
    for x in data:
        if original == x['output']:
            input = x['input']
            break

    
    for key in ['query2', 'query3', 'query4']:
        if key in item and item[key] is not None:
            # conversations = []
            output = item[key]
            
            # human_prompt = input
            assistant_prompt = output
            
            start_index = input.find("Dataset Columns: ")
            question = input[:start_index]
            # print(question)
            start_index = start_index + len("Dataset Columns: ")
            
            dataset_description = input[start_index:]
            # print(dataset_description)
            columns = dataset_description.split('), ')
            formatted_columns = [col + ')' if not col.endswith(')') else col for col in columns]

            # Join the columns with a newline
            formatted_description = '\n'.join(formatted_columns)

            # Print the formatted description
            # print(formatted_description)
            # print("\n\n")
            
            human_prompt = question.rstrip()+" \n "+"Dataset Columns:\n"+formatted_description
            
            conversations = {"instruction": human_prompt, "input": "", "output": assistant_prompt}
            json_obj.append(conversations)
            
for item in final_data:
    # conversations = []
    input = item['input']
    output = item['output']
    
    
    human_prompt = input
    assistant_prompt = output
    
    start_index = input.find("Dataset Columns: ")
    question = input[:start_index]
            # print(question)
    start_index = start_index + len("Dataset Columns: ")
            
    dataset_description = input[start_index:]
            # print(dataset_description)
    columns = dataset_description.split('), ')
    formatted_columns = [col + ')' if not col.endswith(')') else col for col in columns]

            # Join the columns with a newline
    formatted_description = '\n'.join(formatted_columns)

            # Print the formatted description
            # print(formatted_description)
    # print("\n\n")
    
    human_prompt = question.rstrip()+" \n "+"Dataset Columns:\n"+formatted_description
    
    
    
    conversations = {"instruction": human_prompt, "input": "", "output": assistant_prompt}
    json_obj.append(conversations)
    
    for key in ['query2', 'query3', 'query4']:
        if key in item and item[key] is not None:
            # conversations = []
            output = item[key]
            
            assistant_prompt = output
            
            start_index = input.find("Dataset Columns: ")
            question = input[:start_index]
                    # print(question)
            start_index = start_index + len("Dataset Columns: ")
                    
            dataset_description = input[start_index:]
                    # print(dataset_description)
            columns = dataset_description.split('), ')
            formatted_columns = [col + ')' if not col.endswith(')') else col for col in columns]

                    # Join the columns with a newline
            formatted_description = '\n'.join(formatted_columns)

                    # Print the formatted description
                    # print(formatted_description)
            # print("\n\n")
            
            human_prompt = question.rstrip()+" \n "+"Dataset Columns:\n"+formatted_description
            conversations = {"instruction": human_prompt, "input": "", "output": assistant_prompt}
            json_obj.append(conversations)
            
# current_data.extend(json_obj)
random.shuffle(json_obj)

# print(len(current_data))
print(len(json_obj))

with open('train_dev_test/axolotl/transformed/alpaca_trans.json', 'w') as f:
    json.dump(json_obj, f, indent=4)
