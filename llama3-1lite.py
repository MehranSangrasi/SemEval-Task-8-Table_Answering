from llama_cpp import Llama
import pandas as pd

llm = Llama.from_pretrained(
	repo_id="bartowski/Llama-3.1-SuperNova-Lite-GGUF",
	filename="Llama-3.1-SuperNova-Lite-IQ2_M.gguf",
)

# Step 2: Load the dataset
print("Loading dataset from CSV...")
file_path = "table_info.csv"  # Make sure this is the correct path to your file
df = pd.read_csv(file_path)


all_qa = "all_qa.csv"
aq = pd.read_csv(all_qa)


# Step 5: Define a function to query the model
def query_llama(dataset_id, question):

    # Debug: Print the columns of the loaded DataFrame
    # print("Columns in the dataset:", df.columns)

    # Step 3: Prepare the dataset as a text summary
    print("Preparing dataset for Llama 3.1...")

    # Step 2: Ask the user for the Dataset_ID
    # dataset_id = input("Enter the Dataset_ID you want to query (e.g., '001_Forbes'): ")

    # Step 3: Filter the DataFrame for the given Dataset_ID
    filtered_df = df[df['Dataset_ID'] == dataset_id]

    # Check if the Dataset_ID exists
    # Step 4: Check if the Dataset_ID exists and retrieve columns
    # if not filtered_df.empty:
    columns = filtered_df['Columns'].values[0]  # Get the Columns value
    print(columns)

    # Proceed with querying the model...


    # Step 4: Set up the Llama 2 pipeline
    print("Setting up Llama 3.1 pipeline...")
#     prompt = f"""
# I will provide a dataset summary, and you need to generate a python code based on the question so I can run it and find the answer. You need to infer from the column names about what columns will be used to find the answer according to the question and also understand what should be the data type of the answer. Provide nothing else but a dataframe query that is optimized. Make sure to use the column names and their data types as well to understand the question and what the answer data type should be (all possible output data types given below) depending on the question type (what, which, how many, is there, etc.). Just use the columns required to filter the data and find the answer. DONT EXPLAIN. JUST GIVE THE QUERY.

# For example, for the question: "What's the most common gender among the survivors?"

# You should provide the following code:

# df[df['Survived'] == True]['Sex'].mode()[0]


# Output data types:
# 1. list[number]
# 2. number
# 3. boolean
# 4. category
# 5. list[category]

# Dataset form multiple columns of: (column_name);(data_type)
# for_example:  rank; uint16, personName; category, age; float64, finalWorth; uint32

# Dataset summary:
# {columns}

# Question: {question}


# Answer:
# """

    prompt=f"""
I will provide a dataset summary, and you need to generate a python code based on the question. Provide nothing else but the dataframe query python code only.
Do not use more columns than needed to give the answer - give optimized query.. Use the question and find the similarity of the column with it to infer which columns will be used to answer it. Do NOT use any many columns as possible, only limited ones that are sufficient to give the answer.


**Examples:**

Question: "What's the most common gender among the survivors?"

You should provide the following code:
df[df['Survived'] == True]['Sex'].mode()[0]

Question: "How many passengers boarded without any siblings or spouses?"

You should provide the following code: 
df[df['Siblings_Spouses Aboard'] == 0].shape[0]


Example of output types:
how many: 3(int)
which of these is(are) or has(have) : [1,2,3], [steve, bill, elon] (list) or 3.14 (single) note: observe the singularity or multiple value requirement
is there: True/False (bool)
what is: 3.14(single value)
what are: [1,2,3] (list)

All output data types:
1. list[number]
2. number
3. boolean
4. category
5. list[category]

Dataset form multiple columns of: (column_name);(data_type)
for_example:  rank; uint16, personName; category, age; float64, finalWorth; uint32

I will provide the dataset summary and question now, you need to provide the dataframe query.

Dataset summary:
{columns}

Question: {question}

    """

#     prompt=f"""
# You are an expert data assistant. I will provide a dataset summary and a question regarding that dataset. Generate a Python dataframe query that correctly answers the question. The query must strictly adhere to the following guidelines:

# 1. Use only the provided column names and their data types from the dataset summary. Do not make assumptions about extra columns being used, or using more columns than necessary.
# 2. The query must be optimized and directly executable.
# 3. The answer's data type must match the intent of the question, based on these rules:
#    - "how many" → Return a single number (e.g., int or float).
#    - "what is/are" → Return a single value or a list of values (e.g., category, list[category]).
#    - "is there/are there" → Return a boolean (True/False).
#    - "list" → Return a list[number] or list[category].
# 4. I am providing the dataset summary which will include the data type of each column too. You need to use that and the column name to infer which columns will be used to answer the question.

# Output only the dataframe query code. Do not provide explanations or comments.

# ---

# ### Example:

# **Question:**
# What is the most common gender among the survivors?

# **Answer:**
# df[df['Survived'] == True]['Sex'].mode()[0]

# ---

# Now I will provide the dataset summary and the question. You must respond with the Python dataframe query only that is correct.

# Dataset summary:
# {columns}

# Question:
# {question}
# """
    
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response['choices'][0]['message']['content']
# Step 6: Ask questions about the dataset
print("Asking questions about the dataset...")

# Example Question 1: Maximum age of passengers
# question_1 = "Which are the top 4 fares paid by survivors?"
# dataset_1 = "002_Titanic"

# answer1 = query_llama(dataset_1, question_1 )
# # print(f"Question: {question1}")
# print(f"Answer: {answer1}")

results = []

for index, row in aq.head(45).iterrows():
    question = row['question']
    dataset_id = row['dataset']
    
    # Call the query_llama function with extracted values
    print(f"Processing Question {index + 1} with dataset {dataset_id}: {question}")

    answer = query_llama(dataset_id, question)

    print(f"Answer: {answer}\n")
    results.append({'question': question, 'dataset_id': dataset_id, 'answer': answer})


results_df = pd.DataFrame(results)

output_file = "queries_1.csv"
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")


   

# Example Question 2: Average age of passengers
# question2 = "What is the average age of the passengers?"
# answer2 = query_llama(dataset_text, question2)
# print(f"Question: {question2}")
# print(f"Answer: {answer2}")

# # Example Question 3: Total number of survivors
# question3 = "What are the oldest 3 ages among the survivors?"
# answer3 = query_llama(dataset_text, question3)
# print(f"Question: {question3}")
# print(f"Answer: {answer3}")
