import pandas as pd

# Load the data
semeval_train = pd.read_csv("data/semeval_train.csv")
training = pd.read_csv("data/training_dataset.csv")

# Filter rows where the 'question' in semeval_train is not in training['question']
filtered_df = semeval_train[~semeval_train['question'].isin(training['question'])]

# Save the filtered DataFrame to a CSV file
filtered_df.to_csv('data/wrong_queries.csv', index=False)

# print("Filtered CSV has been saved as 'filtered_semeval_train.csv'")
