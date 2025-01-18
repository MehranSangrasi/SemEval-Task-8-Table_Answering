import pandas as pd


prompt = """\nYou are Qwen, a helpful assistant designed for coding. You are supposed to generate single line python pandas data-frame executable query for tabular Q/A given a question and its dataset columns with some columns having unique values.\n\n\n"""
dataset_unique = pd.read_csv("formatted_data.csv")

test_qa = pd.read_csv("competition/competition/test_qa.csv")


conversations = {"conversations": []}

for index, row in test_qa.iterrows():
    matching_columns = dataset_unique.loc[
        dataset_unique["dataset"] == row["dataset"], "columns"
    ]
    
    if not matching_columns.empty:
        # Extract the columns (assuming there's only one match per dataset)
        columns = matching_columns.iloc[0]
        print(f"Dataset: {row['dataset']}")
        print(f"Columns: {columns}")
    else:
        print(f"No matching dataset found for: {row['dataset']}")
        
    question = f"Question: {row['question']} \n Dataset Columns:\n{columns}"
        
    conversations['conversations'].append([{"from": "human", "value": prompt+question}])
    

conversations_df = pd.DataFrame(conversations)

output_csv_path = "test_conversations.csv"
conversations_df.to_csv(output_csv_path, index=False)
print(conversations_df.head())