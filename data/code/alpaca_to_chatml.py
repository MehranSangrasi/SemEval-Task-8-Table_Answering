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
    
system_prompt = """You are Qwen, a helpful assistant designed for coding. You are supposed to generate single line python pandas data-frame executable query for tabular Q/A given a question and its dataset columns with some columns having unique values.

**Examples**:

Question: What are the top 3 most common marital statuses among our employees? \n Dataset Columns: Left (category, unique_values: [Yes, No]), Satisfaction Level (float64), Work Accident (category, unique_values: [Yes, No]), Average Monthly Hours (uint16), Last Evaluation (float64), Years in the Company (uint8, unique_values: [3, 5, 4, 6, 2, 8, 10, 7]), salary (category, unique_values: [low, medium, high]), Department (category), Number of Projects (uint8, unique_values: [2, 5, 4, 6, 7, 3]), Promoted in the last 5 years? (category, unique_values: [Yes, No]), Date Hired (datetime64[ns), UTC], Marital_Status (category, unique_values: [Together, Single, Married])

Query: df.groupby('Marital_Status').size().sort_values(ascending=False).head(3).index.tolist()

Question: Were there any employees hired in 2019? \n Dataset Columns: \nLeft (category, unique_values: [Yes, No]), Satisfaction Level (float64), Work Accident (category, unique_values: [Yes, No]), Average Monthly Hours (uint16), Last Evaluation (float64), Years in the Company (uint8, unique_values: [3, 5, 4, 6, 2, 8, 10, 7]), salary (category, unique_values: [low, medium, high]), Department (category), Number of Projects (uint8, unique_values: [2, 5, 4, 6, 7, 3]), Promoted in the last 5 years? (category, unique_values: [Yes, No]), Date Hired (datetime64[us), UTC]

Query: pd.to_datetime(df['Date Hired']).dt.year.eq(2019).any()

**Now the question:**
"""

json_obj = []

for item in data:
    conversations = {"conversations": [], "tools": "[]"}
    input = item['input']
    output = item['output']
    # human_prompt = system_prompt + "\n\n\n" + input
    human_prompt = input
    assistant_prompt = output
    
    conversation1 = {'from': 'human', 'value': human_prompt}
    conversation2 = {'from': 'gpt', 'value': assistant_prompt}
    conversations["conversations"].append(conversation1)
    conversations["conversations"].append(conversation2)
    json_obj.append(conversations)
    
for item in other_data['messages']:

    original = item['output']
    
    for x in data:
        if original == x['output']:
            input = x['input']
            break

    
    for key in ['query2', 'query3', 'query4']:
        if key in item and item[key] is not None:
            conversations = {"conversations": [], "tools": "[]"}
            output = item[key]
            
            human_prompt = input
            assistant_prompt = output
            conversation1 = {'from': 'human', 'value': human_prompt}
            conversation2 = {'from': 'gpt', 'value': assistant_prompt}
            conversations["conversations"].append(conversation1)
            conversations["conversations"].append(conversation2)
            json_obj.append(conversations)
            
for item in final_data:
    conversations = {"conversations": [], "tools": "[]"}
    input = item['input']
    output = item['output']
    
    human_prompt = input
    assistant_prompt = output
    conversation1 = {'from': 'human', 'value': human_prompt}
    conversation2 = {'from': 'gpt', 'value': assistant_prompt}
    conversations["conversations"].append(conversation1)
    conversations["conversations"].append(conversation2)
    json_obj.append(conversations)
    
    for key in ['query2', 'query3', 'query4']:
        if key in item and item[key] is not None:
            conversations = {"conversations": [], "tools": "[]"}
            output = item[key]
            
            human_prompt = system_prompt + "\n\n\n" + input
            assistant_prompt = output
            conversation1 = {'from': 'human', 'value': human_prompt}
            conversation2 = {'from': 'gpt', 'value': assistant_prompt}
            conversations["conversations"].append(conversation1)
            conversations["conversations"].append(conversation2)
            json_obj.append(conversations)
            
# current_data.extend(json_obj)
random.shuffle(json_obj)

# print(len(current_data))
print(len(json_obj))

with open('data/train_transformed.json', 'w') as f:
    json.dump(json_obj, f, indent=4)
