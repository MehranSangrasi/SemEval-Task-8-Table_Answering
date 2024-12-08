from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
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
def query_gpt4(dataset_id, question):
    # Filter the DataFrame for the given Dataset_ID
    filtered_df = df[df['Dataset_ID'] == dataset_id]

    if filtered_df.empty:
        return f"Dataset_ID {dataset_id} not found."

    # Get the Columns value
    columns = filtered_df['Columns'].values[0]
    print(f"Dataset Columns: {columns}")

    # Prepare the prompt
    prompt = f"""
    I will provide a dataset columns and their data types as well, and you need to generate a python code as a dataframe query based on the question. Provide nothing else but the dataframe query python code only.
    Do not use more columns than needed to give the answer - give optimized query. Use the question and find the similarity of the column with it to infer which columns will be used to answer it. Do NOT use any more columns than necessary. You need to infer from the question that which output data type is needed from the ones provided below. You can use the examples below to understand the output data types. You need to provide the query to get the answer from the dataset, also make sure that the columns you use are present in the dataset columns that I provide. Make sure to use as minimum columns as possible and give an optimized correct query that will answer the question by inferring from the question and the dataset columns.
    
    **Note**: You will not get what is the output data type in the question itself, but you can analyze the examples below in which I have mentioned the output data type in order to learn for the question I ask. You will STRICTLY not get it when I give you the question myself. Do not include ```python ``` tags in your output.


    All output data types:
    1. list[number]
    2. number
    3. boolean
    4. category
    5. list[category]
    
    
    **Examples for each output data type:**

    Question: "What's the country of origin of the oldest billionaire?" - output data type: category
    Dataset summary: rank; uint16, personName; category, age; float64, finalWorth; uint32, category; category, source; category, country; category, state; category, city; category, organization; category, selfMade; bool, gender; category, birthDate; datetime64[us, UTC], title; category, philanthropyScore; float64, bio; object, about; object
    Output: df.loc[df['age'] == df['age'].max(), 'country'].iloc[0]

    Question: "Which are the top 4 fares paid by survivors?" - output data type: list[number] / list of numbers
    Dataset summary: Survived; bool, Pclass; uint8, Name; object, Sex; category, Age; float64, Siblings_Spouses Aboard; uint8, Parents_Children Aboard; uint8, Fare; float64
    Output: df[df['Survived'] == True].nlargest(4, 'Fare')['Fare'].tolist()
    
    Question: "Do the majority of respondents have a height greater than 170 cm?" - output data type: boolean
    Dataset summary: Submitted at; datetime64[us, UTC], What is your age? 👶🏻👵🏻; uint8, What's your nationality?; category, What is your civil status? 💍; category, What's your sexual orientation?; category, Do you have children? 🍼; category, What is the maximum level of studies you have achieved? 🎓; category, Gross annual salary (in euros) 💸; float64, What's your height? in cm 📏; uint8, What's your weight? in Kg ⚖️; float64, What is your body complexity? 🏋️; category, What is your eye color? 👁️; category, What is your hair color? 👩🦰👱🏽; category, What is your skin tone?; uint8, How long is your hair? 💇🏻♀️💇🏽♂️; category, How long is your facial hair? 🧔🏻; category, How often do you wear glasses? 👓; category, How attractive do you consider yourself?; uint8, Have you ever use an oline dating app?; category, Where have you met your sexual partners? (In a Bar or Restaurant); bool, Where have you met your sexual partners? (Through Friends); bool, Where have you met your sexual partners? (Through Work or as Co-Workers); bool, Where have you met your sexual partners? (Through Family); bool, Where have you met your sexual partners? (in University); bool, Where have you met your sexual partners? (in Primary or Secondary School); bool, Where have you met your sexual partners? (Neighbors); bool, Where have you met your sexual partners? (in Church); bool, Where have you met your sexual partners? (Other); bool, How many people have you kissed?; uint16, How many sexual partners have you had?; uint16, How many people have you considered as your boyfriend_girlfriend?; uint8, How many times per month did you practice sex lately?; float64, Happiness scale; uint8, What area of knowledge is closer to you?; object, If you are in a relationship, how long have you been with your partner?; float64
    Output: df['What\'s your height? in cm 📏'] > 170).mean() > 0.5
    
    Question: "Which are the top 5 nationalities in terms of the average overall score of their players?" - output data type: list[category] or list of categories
    Dataset summary: ID<gx:number>; uint32, Name<gx:text>; category, Age<gx:number>; uint8, Photo<gx:url>; category, Nationality<gx:category>; category, Overall<gx:number>; uint8, Potential<gx:number>; uint8, Club<gx:category>; category, Value_€<gx:currency>; uint32, Wage_€<gx:currency>; uint32, Preferred Foot<gx:category>; category, International Reputation<gx:number>; uint8, Weak Foot<gx:number>; uint8, Skill Moves<gx:number>; uint8, Work Rate<gx:category>; category, Position<gx:category>; category, Joined<gx:date>; category, Contract Valid Until<gx:date>; category, Height_ft<gx:number>; float64, Weight_lbs<gx:number>; uint8, Crossing<gx:number>; uint8, Finishing<gx:number>; uint8, HeadingAccuracy<gx:number>; uint8, ShortPassing<gx:number>; uint8, Volleys<gx:number>; uint8, Dribbling<gx:number>; uint8, Curve<gx:number>; uint8, FKAccuracy<gx:number>; uint8, LongPassing<gx:number>; uint8, BallControl<gx:number>; uint8, Acceleration<gx:number>; uint8, SprintSpeed<gx:number>; uint8, Agility<gx:number>; uint8, Reactions<gx:number>; uint8, Balance<gx:number>; uint8, ShotPower<gx:number>; uint8, Jumping<gx:number>; uint8, Stamina<gx:number>; uint8, Strength<gx:number>; uint8, LongShots<gx:number>; uint8, Aggression<gx:number>; uint8, Interceptions<gx:number>; uint8, Positioning<gx:number>; uint8, Vision<gx:number>; uint8, Penalties<gx:number>; uint8, Composure<gx:number>; uint8, Marking<gx:number>; category, StandingTackle<gx:number>; uint8, SlidingTackle<gx:number>; uint8, GKDiving<gx:number>; uint8, GKHandling<gx:number>; uint8, GKKicking<gx:number>; uint8, GKPositioning<gx:number>; uint8, GKReflexes<gx:number>; uint8, Best Position<gx:category>; category, Best Overall Rating<gx:number>; uint8, DefensiveAwareness<gx:number>; uint8, General Postion<gx:category>; category, Legend; bool
    
    Output: df.groupby('Nationality<gx:category>')['Overall<gx:number>'].mean().nlargest(5).index.tolist()
    
    Question: "How many unique countries gave 'Wise' a rating of 5?" - output data type: list[number] / list of numbers
    Dataset summary: published_date; datetime64[us, UTC], country_code; category, title; category, body; object, rating; uint8, Company; category
    Output: df[(df['Company'] == 'Wise') & (df['rating'] == 5)]['country_code'].nunique()
    
    

    **Now for the question:**

    Dataset columns:
    {columns}

    Question: {question}
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
    
    columns = df[df['Dataset_ID'] == dataset_id]['Columns']
    
    print(f"Processing Question {index + 1} with dataset {dataset_id}: {question}")
    
    try:
        answer = query_gpt4(dataset_id, question)
        # print(answer)
        print(f"Answer: {answer}\n")
        results.append({'question': question, 'dataset_id': dataset_id, 'columns': columns, 'query': answer})
    except Exception as e:
        print(f"Error processing Question {index + 1}: {str(e)}")
        results.append({'question': question, 'dataset_id': dataset_id, 'columns': columns, 'query': f"Error: {str(e)}"})

# Save results to a CSV file
results_df = pd.DataFrame(results)
output_file = "data/queries_final_2.csv"
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
