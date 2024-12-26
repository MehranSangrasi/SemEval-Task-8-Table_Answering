from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import json
import os
import requests
import time
from datasets import load_dataset

# **Example of output types:**
#     how many: 3 (int)
#     which of these is(are) or has(have): [1,2,3], [steve, bill, elon] (list) or 3.14 (single)
#     is there: True/False (bool)
#     what is: 3.14 (single value) or (category)
#     what are: [1,2,3] (list)


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
OpenAI.api_key = api_key

# Load the dataset
print("Loading dataset from CSV...")
file_path = "data/pointless/table_info.csv"  # Make sure this is the correct path to your file
semeval = load_dataset("cardiffnlp/databench", name="semeval", split="train")
aq = semeval.to_pandas()
# print(aq)
df = pd.read_csv(file_path)

# all_qa = "all_qa.csv"
# aq = pd.read_csv(all_qa)

# Function to query GPT-4 API
def query_gpt4(dataset_id, question, sample_answer, answer_type,columns_used, column_types):
    # Filter the DataFrame for the given Dataset_ID
    filtered_df = df[df['Dataset_ID'] == dataset_id]

    if filtered_df.empty:
        return f"Dataset_ID {dataset_id} not found."

    # Get the Columns value
    columns = filtered_df['Columns'].values[0]
    
    # print(f"Dataset Columns: {columns}")
    
    sample_answer = sample_answer
    answer_type = answer_type
    columns_used = columns_used
    column_types = column_types
    
    
    
    
    

    # Prepare the prompt
    prompt = f"""
    I will provide a dataset columns and their data types as well, and you need to generate a python code as a dataframe query based on the question. Provide nothing else but the dataframe query python code only.
    Do not use more columns than needed to give the answer - give optimized query. Use the questions and the columns used provided to you to generate the query. Do not use any more columns than necessary as I have provided you which columns have been used to give actual answer. Strictly make sure to use as optimal columns as possible and give an optimized correct query that will answer the question by inferring from the question and the dataset columns. You have been provided the columns that have been used to answer the question and their data types. You need to provide the query to get the answer from the dataset. Do not add any additional explanations or output, just the query. Make sure that the output that the query gives is the same as the actual answer provided to you by using the columns_used and columns_used_types.
    
    **Note**: I am giving you the dataset columns, their data types, the question, the actual answer, answer type, columns_used and columns_used_types. You need to provide the query to get the answer from the dataset. Do not use any ```python ``` or ```pandas``` code, just the query to get the answer from the dataset.
    
    Generate a query ONLY to get the answer after reading the question, dataset columns, and the actual answer provided to you. Do not use any additional columns than the ones provided to you. You need to provide the query to get the answer from the dataset. Make sure that the output that the query gives is the same as the actual answer provided to you.

    All output data types:
    1. list[number]
    2. number
    3. boolean
    4. category
    5. list[category]
    
    
    **Examples for each output data type:**
    
    "question": "What's the country of origin of the oldest billionaire?",
        "dataset_id": "001_Forbes",
        "columns": ["Columns
            "rank; uint16, personName; category, age; float64, finalWorth; uint32, category; category, source; category, country; category, state; category, city; category, organization; category, selfMade; bool, gender; category, birthDate; datetime64[us, UTC], title; category, philanthropyScore; float64, bio; object, about; object"],
        "sample_answer": United States,
        "answer_type": category,
        "columns_used": ['age', 'country'],
        "columns_used_types": ['number[UInt8]', 'category'],
        "query": df.loc[df['age'] == df['age'].max(), 'country'].iloc[0],
        
        
     "question": "Which are the top 4 fares paid by survivors?",
        "dataset_id": "002_Titanic",
        "columns": ["Columns
           "Survived; bool, Pclass; uint8, Name; object, Sex; category, Age; float64, Siblings_Spouses Aboard; uint8, Parents_Children Aboard; uint8, Fare; float64"
        "sample_answer": [512.239, 512.239, 512.239, 263],
        "answer_type": list[number],
        "columns_used": [Fare, Survived],
        "columns_used_types": ['number[double]', 'boolean']  
        "query": df[df['Survived'] == True].nlargest(4, 'Fare')['Fare'].tolist(),

    
    
    

    "question": "Do the majority of respondents have a height greater than 170 cm?",
        "dataset_id": "003_Love",
        "columns": ["Columns
            Submitted at; datetime64[us, UTC], What is your age? 👶🏻👵🏻; uint8, What's your nationality?; category, What is your civil status? 💍; category, What's your sexual orientation?; category, Do you have children? 🍼; category, What is the maximum level of studies you have achieved? 🎓; category, Gross annual salary (in euros) 💸; float64, What's your height? in cm 📏; uint8, What's your weight? in Kg ⚖️; float64, What is your body complexity? 🏋️; category, What is your eye color? 👁️; category, What is your hair color? 👩🦰👱🏽; category, What is your skin tone?; uint8, How long is your hair? 💇🏻♀️💇🏽♂️; category, How long is your facial hair? 🧔🏻; category, How often do you wear glasses? 👓; category, How attractive do you consider yourself?; uint8, Have you ever use an oline dating app?; category, Where have you met your sexual partners? (In a Bar or Restaurant); bool, Where have you met your sexual partners? (Through Friends); bool, Where have you met your sexual partners? (Through Work or as Co-Workers); bool, Where have you met your sexual partners? (Through Family); bool, Where have you met your sexual partners? (in University); bool, Where have you met your sexual partners? (in Primary or Secondary School); bool, Where have you met your sexual partners? (Neighbors); bool, Where have you met your sexual partners? (in Church); bool, Where have you met your sexual partners? (Other); bool, How many people have you kissed?; uint16, How many sexual partners have you had?; uint16, How many people have you considered as your boyfriend_girlfriend?; uint8, How many times per month did you practice sex lately?; float64, Happiness scale; uint8, What area of knowledge is closer to you?; object, If you are in a relationship, how long have you been with your partner?; float64
        "actual_answer": True,
        "answer_type": boolean,
        "columns_used": ['What's your height? in cm 📏'],
        "columns_used_types": "['number[UInt8]']",
        "query": "df['What's your height? in cm 📏'] > 170).mean() > 0.5",
    
    "question": "Which are the top 5 nationalities in terms of the average overall score of their players?",
        "dataset_id": "007_FIFA",
        "columns": ["Columns
             "ID<gx:number>; uint32, Name<gx:text>; category, Age<gx:number>; uint8, Photo<gx:url>; category, Nationality<gx:category>; category, Overall<gx:number>; uint8, Potential<gx:number>; uint8, Club<gx:category>; category, Value_€<gx:currency>; uint32, Wage_€<gx:currency>; uint32, Preferred Foot<gx:category>; category, International Reputation<gx:number>; uint8, Weak Foot<gx:number>; uint8, Skill Moves<gx:number>; uint8, Work Rate<gx:category>; category, Position<gx:category>; category, Joined<gx:date>; category, Contract Valid Until<gx:date>; category, Height_ft<gx:number>; float64, Weight_lbs<gx:number>; uint8, Crossing<gx:number>; uint8, Finishing<gx:number>; uint8, HeadingAccuracy<gx:number>; uint8, ShortPassing<gx:number>; uint8, Volleys<gx:number>; uint8, Dribbling<gx:number>; uint8, Curve<gx:number>; uint8, FKAccuracy<gx:number>; uint8, LongPassing<gx:number>; uint8, BallControl<gx:number>; uint8, Acceleration<gx:number>; uint8, SprintSpeed<gx:number>; uint8, Agility<gx:number>; uint8, Reactions<gx:number>; uint8, Balance<gx:number>; uint8, ShotPower<gx:number>; uint8, Jumping<gx:number>; uint8, Stamina<gx:number>; uint8, Strength<gx:number>; uint8, LongShots<gx:number>; uint8, Aggression<gx:number>; uint8, Interceptions<gx:number>; uint8, Positioning<gx:number>; uint8, Vision<gx:number>; uint8, Penalties<gx:number>; uint8, Composure<gx:number>; uint8, Marking<gx:number>; category, StandingTackle<gx:number>; uint8, SlidingTackle<gx:number>; uint8, GKDiving<gx:number>; uint8, GKHandling<gx:number>; uint8, GKKicking<gx:number>; uint8, GKPositioning<gx:number>; uint8, GKReflexes<gx:number>; uint8, Best Position<gx:category>; category, Best Overall Rating<gx:number>; uint8, DefensiveAwareness<gx:number>; uint8, General Postion<gx:category>; category, Legend; bool"
        "actual_answer": ['Tanzania', 'Syria', 'Mozambique', 'Chad', 'Central African Rep.'],
        "answer_type": list[category],
        "columns_used": ['Nationality<gx:category>', 'Overall<gx:number>'],
        "columns_used_types": ['category', 'number[uint8]'],
        "query": df.groupby('Nationality<gx:category>')['Overall<gx:number>'].mean().nlargest(5).index.tolist(),
        
    "question": "How many unique countries gave 'Wise' a rating of 5?",
        "dataset_id": "031_Trustpilot",
        "columns": ["Columns
            "published_date; datetime64[us, UTC], country_code; category, title; category, body; object, rating; uint8, Company; category"
        "actual_answer": 120,
        "answer_type": number,
        "columns_used": ['country_code', 'Company', 'rating'],
        "columns_used_types": ['category', 'category', 'number[uint8]'],
        "query": df[(df['Company'] == 'Wise') & (df['rating'] == 5)]['country_code'].nunique(),
    
    
    
    
    
    

    **Now for the question with the actual answer, generate the query that provides this answer with respect to decimal places in numbers. The query should give the same answer as provided to you in actual_answer:**

    Total dataset columns:
    {columns}

    Question: {question}
    
    
    actual_answer: {sample_answer}
    answer_type: {answer_type}
    columns_used: {columns_used}
    columns_used_types: {column_types}
    
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


# Iterate through questions and datasets
print("Asking questions about the dataset...")

results = []

for index, row in aq.iterrows():
    question = row['question']
    dataset_id = row['dataset']
    sample_answer = row["answer"]
    answer_type=row["type"]
    columns_used=row["columns_used"]
    column_types=row["column_types"]
    
    # import pdb; pdb.set_trace()
    
    # print(question, dataset_id, sample_answer, answer_type, columns_used, column_types)
    
    columns = df[df['Dataset_ID'] == dataset_id]['Columns']
    
    
    print(f"Processing Question {index + 1} with dataset {dataset_id}: {question}")
    
    try:
        answer = query_gpt4(dataset_id, question, sample_answer, answer_type,columns_used, column_types)
        print(answer)
        # print(f"Answer: {answer}\n")
        # answer_dict = json.loads(answer)
        

# Now you can access the query like this:
        query = answer
        
        results.append({'question': question, 'dataset_id': dataset_id, 'columns': columns, 'query': query})

    except Exception as e:
        print(f"Error processing Question {index + 1}: {str(e)}")
        results.append({'question': question, 'dataset_id': dataset_id, 'columns': columns, 'query': f"Error: {str(e)}"})

# Save results to a CSV file
results_df = pd.DataFrame(results)
output_file = "data/queries_new.csv"
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
