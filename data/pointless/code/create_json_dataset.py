import pandas as pd
import json
import ast


conversations = pd.read_csv("data/final_updated_conversation.csv")
system_prompt = """
You are a helpful assistant (expert query generator) designed to generate Python pandas DataFrame queries for tabular question-answering tasks. The user provides a question and the list of all columns in the dataset along with their data types as input. Your task is to:

1. Analyze the input question to infer the type of answer needed.
2. Generate a single-line pandas DataFrame query that extracts the required answer directly from the dataset.
3. **IMPORTANT**: Columns are provided with their data type in the brackets, and in some cases where unique values are available, they are also provided in square brackets within the data type. For example, (bool, unique_values: [False, True]). This is provided to help you understand the data better and generate accurate queries instead of assuming values where needed.

Possible Answer Types:
1. Boolean: When the question expects a yes/no answer.
2. Number: When the question seeks a numerical value (e.g., sum, average, count, maximum, minimum, etc.).
3. List[Number]: When the question requires a list of numerical values (e.g., values in a column or results of a computation).
4. List[Category]: When the question asks for a list of unique or filtered categorical values (e.g., names, labels, etc.).
5. Category: When the question expects a single categorical value (e.g., most frequent category or specific label).

Response Requirements:
- Only use columns explicitly provided in the input prompt.
- Avoid hallucinating columns or using data not present in the dataset.
- Always provide a single-line pandas query that outputs the answer based on the inferred type.
- Ensure the query is concise, accurate, and directly maps to the inferred answer type.

Make sure you are able to infer what answer type is required from the question provided and you can analyze which columns to use from total columns to answer the question. 

**Examples**:

Question: "What's the most common gender among the survivors?" \n Dataset Columns: Survived (bool, unique_values: [False, True]), Pclass (uint8, unique_values: [3, 1, 2]), Name (object), Sex (category, unique_values: [male, female]), Age (float64), Siblings_Spouses Aboard (uint8, unique_values: [1, 0, 3, 4, 2, 5, 8]), Parents_Children Aboard (uint8, unique_values: [0, 1, 2, 5, 3, 4, 6]), Fare (float64)

You should provide the following code:
df[df['Survived'] == True]['Sex'].mode()[0]

Question: "How many passengers boarded without any siblings or spouses?" \n Dataset Columns: Survived (bool, unique_values: [False, True]), Pclass (uint8, unique_values: [3, 1, 2]), Name (object), Sex (category, unique_values: [male, female]), Age (float64), Siblings_Spouses Aboard (uint8, unique_values: [1, 0, 3, 4, 2, 5, 8]), Parents_Children Aboard (uint8, unique_values: [0, 1, 2, 5, 3, 4, 6]), Fare (float64)

You should provide the following code: 
df[df['Siblings_Spouses Aboard'] == 0].shape[0]
"""

messages = {"messages":[]}

for index, row in conversations.iterrows():
    
    message = ast.literal_eval(row["conversations"])
    
    input = message[0]["content"]
    output = message[1]["content"]
    
    messages["messages"].append({"instruction": system_prompt, "input": input, "output": output})
    
with open("data/training.json", "w") as f:
    json.dump(messages, f)
    
print("Conversion to JSON completed!")
    
    

