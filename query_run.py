import pandas as pd

# Define your DataFrames (example data)
titanic = pd.read_csv("titanic.csv")
forbes = pd.read_csv("Forbes.csv")

titanic_df = pd.DataFrame(titanic)
forbes_df = pd.DataFrame(forbes)
# A dictionary mapping DataFrame names to actual DataFrames
dataframes = {
    "002_Titanic": titanic_df, 
    "001_Forbes": forbes_df,
}

# Load the CSV file containing queries and DataFrame names
queries_csv_path = 'queries_1.csv'
queries_df = pd.read_csv(queries_csv_path)

# Assume the columns are 'dataframe_name' and 'query'
if 'dataset_id' not in queries_df.columns or 'answer' not in queries_df.columns:
    raise ValueError("The CSV file must contain 'dataframe_name' and 'query' columns.")

# Execute each query on the specified DataFrame
for idx, row in queries_df.iterrows():
    dataframe_name = row['dataset_id']
    query = row['answer']

    # Check if the DataFrame exists
    if dataframe_name not in dataframes:
        print(f"Error: DataFrame '{dataframe_name}' not found for Query {idx + 1}.")
        continue

    # Get the DataFrame
    target_df = dataframes[dataframe_name]

    try:
        # Execute the query
        result = target_df.query(query)
        print(f"Result for Query {idx + 1} (DataFrame: {dataframe_name}):")
        print(result)
    except Exception as e:
        print(f"Error executing Query {idx + 1} on DataFrame '{dataframe_name}': {query}")
        print(f"Error: {e}")
