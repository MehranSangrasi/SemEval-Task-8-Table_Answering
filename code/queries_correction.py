from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import json
import os
import requests
import time


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
OpenAI.api_key = api_key

df = pd.read_csv("data/wrong_queries.csv")
table_info = pd.read_csv("data/table_info.csv")


# Function to query GPT-4 API
def query_gpt4(dataset_id, question, answer, answer_type, columns_used, column_types, columns_total, query=''):
    
    answer = answer
    answer_type = answer_type
    columns_used = columns_used
    column_types = column_types
    columns_total = columns_total
    

    # Prepare the prompt
    system_prompt = """
    
    You are a helpful assistant tasked with generating the right query to answer the question from a dataset when given its columns. You will be provided with the question, the dataset columns, the actual answer of the question, the data type of the answer, the columns used to generate the answer, and the data types of the columns used. You need to generate the right query to answer the question using the dataset columns. 
    
    Generate a python dataframe pandas query to answer the question using the columns used from total columns.
    
    Make sure to **strictly** generate the right query. Give only the query, i dont want any variable names or ```python ``` for language definition of the code snippet. Only the query like:
    
    df.loc[df['finalWorth'].idxmax(), 'selfMade']

    The answer should be exactly the same like this without any addition. 
    DO NOT ADD '```python```' OR '```' AT THE START OR END OF THE QUERY. ONLY THE QUERY.
    Make sure there is df in the code as it represents the dataset.
    
    """
    
    prompt = f"""

    
    The answer should be exactly the same like this without any addition.
    STRICT INSTRUCTIONS:
    1. **If there is no query in the prompt, that means you are supposed to generate the query yourself.**
    2. **If there is query present in the prompt, that means you are supposed to correct it and generate the right query.**
    Strictly dont add '```python' or '```' at the start or end of the query. Only the query.
    
    Question: {question}
    Actual_Answer: {answer}
    Answer_Type: {answer_type}
    Columns_Used: {columns_used}
    Column_Types: {column_types}
    Columns_Total: {columns_total}
    Query: {query}
    
    Output the right query using the columns from columns used:
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Prepare the payload for OpenAI API request
    payload = {
        "model": "gpt-4o",  # Use appropriate model like gpt-4, gpt-3.5-turbo, etc.
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 4000,  # Adjust token limit as per your requirement
        "temperature": 0.5  # Adjust temperature for more or less creative output
    }

    retries = 3  # Number of retries in case of failure
    for attempt in range(retries):
        try:
            # Make the API request to OpenAI
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            
            # Check for a successful response
            if response.status_code == 200:
                response_data = response.json()
                response_query = response_data['choices'][0]['message']['content']
                
                
                return response_query

            else:
                print(f"Attempt {attempt + 1}: Failed to generate notes (Status Code: {response.status_code}). Retrying...")
                print(response.text)
                time.sleep(1)  # Optional delay before retrying

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(1)  # Optional delay before retrying
