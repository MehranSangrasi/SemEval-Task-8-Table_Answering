import pandas as pd
import ast

# Load the CSV file into a DataFrame
file_path = 'data/eval_conversations.csv'  # Replace with your file path
df = pd.read_csv(file_path)
eval_dataset = pd.read_csv("data/dev_set.csv")

df['dataset_name'] = None


for index, row in df.iterrows():
    conversation = ast.literal_eval(row['conversations'])
    question = conversation['content'].split("\n")[0]
    question = question.strip("Question: ")
    question = question.rstrip()
    
    needed = eval_dataset.loc[eval_dataset['question'] == question]["dataset"].values[0]
    
    if needed:
        df.at[index, 'dataset_name'] = needed
        
        
df.to_csv("data/updated_eval_conversations.csv", index=False)

    
    
    
    
    
    
    
    
    
    

# question = df['content'].str.extract(r'Question:\s*(.*?)\s*\\n')

# Display or save the extracted questions
# print(df['question'])
