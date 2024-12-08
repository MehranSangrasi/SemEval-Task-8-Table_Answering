import pandas as pd


queries_file = pd.read_csv('data/queries_final_2.csv')
table_info = pd.read_csv('data/pointless/table_info.csv')

for index, row in queries_file.iterrows():
    dataset_id = row['dataset_id']
    
    columns = table_info[table_info['Dataset_ID'] == dataset_id]['Columns'].values[0]
    
    queries_file.at[index, 'columns'] = columns
    
    # print(f"Question {index + 1} with dataset {dataset_id}: {question}")
    # print(f"Answer: {answer}\n")
    
queries_file.to_csv('data/queries_final_3.csv', index=False)