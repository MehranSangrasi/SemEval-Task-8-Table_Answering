import pandas as pd


table_info = pd.read_csv("/Users/mehran/CodeSpaces/Testing/Table_Answering/data/table_info.csv")

matched_data = pd.read_csv("/Users/mehran/CodeSpaces/Testing/Table_Answering/data/matched_data_2.csv")


# # replace column 'columns' in matched_data with 'Columns' in table_info matched the key 'dataset_id' from matched_data to 'Dataset_ID' in table_info
# matched_data['columns'] = matched_data['dataset_id'].map(table_info.set_index('Dataset_ID')['Columns'])


# check if the columns in the matched_data 'query' have string quotes, if so, remove them
for index, row in matched_data.iterrows():
    # if '"' in matched_data['query'].iloc[index]:
    #     temp = matched_data['query'].iloc[index][1:-1]
    #     matched_data['query'].iloc[index] = temp
    print(row['query'])


# save the new matched_data to a new csv file

matched_data.to_csv("/Users/mehran/CodeSpaces/Testing/Table_Answering/data/matched_data_3.csv", index=False)