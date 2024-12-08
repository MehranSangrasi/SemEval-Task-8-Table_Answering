import pandas as pd

# Load the target DataFrames
titanic = pd.read_csv("titanic.csv")
forbes = pd.read_csv("Forbes.csv")

# Mapping DataFrame names to actual DataFrames
dataframes = {
    "002_Titanic": titanic,
    "001_Forbes": forbes,
}

# Load the CSV file containing queries and DataFrame names
queries_csv_path = 'queries_1.csv'
queries_df = pd.read_csv(queries_csv_path)

# Check required columns in the CSV
required_columns = ['dataset_id', 'answer']
if not all(col in queries_df.columns for col in required_columns):
    raise ValueError(f"The CSV file must contain the columns: {required_columns}")

# List to store results
results = []

# Execute each query
for idx, row in queries_df.iterrows():
    dataframe_name = row['dataset_id']
    query = row['answer']

    # Initialize result as "ERROR" by default
    result_str = "ERROR"

    # Check if the DataFrame exists
    if dataframe_name in dataframes:
        target_df = dataframes[dataframe_name]
        try:
            # Use eval with 'df' as the DataFrame variable
            result = eval(query, {'df': target_df})
            # Convert result to a string for storing
            if isinstance(result, pd.DataFrame):
                result_str = result.to_string(index=False)
            else:
                result_str = str(result)
        except Exception as e:
            result_str = "ERROR"
    else:
        print(f"Error: DataFrame '{dataframe_name}' not found for Query {idx + 1}.")

    # Append result to the results list
    results.append({"S.no": idx + 1, "answers": result_str})

# Save results to a CSV file
output_csv_path = "query_results.csv"
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
