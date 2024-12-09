import pandas as pd

# Load the dataset
df = pd.read_csv('data/queries_with_answers_final_4.csv')  # Replace with your actual file path

# Filter out rows where the 'answers' column has 'ERROR'
filtered_df = df[df['answers'] != 'ERROR']

# Save the filtered dataframe to a new CSV
filtered_df.to_csv('data/filtered_dataset.csv', index=False)

print("Filtered dataset has been saved to 'filtered_dataset.csv'")
