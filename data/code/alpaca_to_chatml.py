import json

# with open('data/train.json') as f:
#     data = json.load(f)
    
# with open("data/updated_queries.json") as f:
#     other_data = json.load(f)

with open('data/chatml_69.json') as f:
    current_data = json.load(f)
    
with open("data/updated_queries_transformation.json") as f:
    final_data = json.load(f)
    
system_prompt = """You are DeepSeek, a helpful assistant designed for coding. You are supposed to generate single line python pandas data-frame executable query for tabular Q/A given a question and its dataset columns with some columns having unique values.

**Examples**:

Question: What are the highest three numbers of followers count present in the dataset? \n Dataset Columns: id<gx:category>; int64, author_id<gx:category>; uint32, author_name<gx:category>; category, author_handler<gx:category>; category, author_avatar<gx:url>; category, user_created_at<gx:date>; datetime64[us, UTC], user_description<gx:text>; category, user_favourites_count<gx:number>; uint16, user_followers_count<gx:number>; uint32, user_following_count<gx:number>; uint16, user_listed_count<gx:number>; uint16, user_tweets_count<gx:number>; uint16, user_verified<gx:boolean>; bool, user_location<gx:text>; category, lang<gx:category>; category, type<gx:category>; category, text<gx:text>; object, date<gx:date>; datetime64[us, UTC], mention_ids<gx:list[category]>; object, mention_names<gx:list[category]>; object, retweets<gx:number>; uint32, favorites<gx:number>; uint32, replies<gx:number>; uint16, quotes<gx:number>; uint16, links<gx:list[url]>; object, links_first<gx:url>; category, image_links<gx:list[url]>; object, image_links_first<gx:url>; category, rp_user_id<gx:category>; category, rp_user_name<gx:category>; category, location<gx:text>; category, tweet_link<gx:url>; category, source<gx:text>; category, search<gx:category>; category

Query: df['user_followers_count<gx:number>'].sort_values(ascending=False).head(3).tolist()

Question: Were there any employees hired in 2019? \n Dataset Columns: Left (category, unique_values: [Yes, No]), Satisfaction Level (float64), Work Accident (category, unique_values: [Yes, No]), Average Monthly Hours (uint16), Last Evaluation (float64), Years in the Company (uint8, unique_values: [3, 5, 4, 6, 2, 8, 10, 7]), salary (category, unique_values: [low, medium, high]), Department (category), Number of Projects (uint8, unique_values: [2, 5, 4, 6, 7, 3]), Promoted in the last 5 years? (category, unique_values: [Yes, No]), Date Hired (datetime64[us), UTC]

Query: pd.to_datetime(df['Date Hired']).dt.year.eq(2019).any()

**Now the question:**
"""

json_obj = []

# for item in data:
#     conversations = {"conversations": [], "tools": "[]"}
#     input = item['input']
#     output = item['output']
#     human_prompt = system_prompt + "\n\n\n" + input
#     assistant_prompt = output
    
#     conversation1 = {'from': 'human', 'value': human_prompt}
#     conversation2 = {'from': 'gpt', 'value': assistant_prompt}
#     conversations["conversations"].append(conversation1)
#     conversations["conversations"].append(conversation2)
#     json_obj.append(conversations)
    
# for item in other_data['messages']:

#     original = item['output']
    
#     for x in data:
#         if original == x['output']:
#             input = x['input']
#             break

    
#     for key in ['query2', 'query3', 'query4']:
#         if key in item and item[key] is not None:
#             conversations = {"conversations": [], "tools": "[]"}
#             output = item[key]
            
#             human_prompt = system_prompt + "\n\n\n" + input
#             assistant_prompt = output
#             conversation1 = {'from': 'human', 'value': human_prompt}
#             conversation2 = {'from': 'gpt', 'value': assistant_prompt}
#             conversations["conversations"].append(conversation1)
#             conversations["conversations"].append(conversation2)
#             json_obj.append(conversations)
            
for item in final_data:
    conversations = {"conversations": [], "tools": "[]"}
    input = item['input']
    output = item['output']
    
    human_prompt = system_prompt + "\n\n\n" + input
    assistant_prompt = output
    conversation1 = {'from': 'human', 'value': human_prompt}
    conversation2 = {'from': 'gpt', 'value': assistant_prompt}
    conversations["conversations"].append(conversation1)
    conversations["conversations"].append(conversation2)
    json_obj.append(conversations)
    
    for key in ['query2', 'query3', 'query4']:
        if key in item and item[key] is not None:
            conversations = {"conversations": [], "tools": "[]"}
            output = item[key]
            
            human_prompt = system_prompt + "\n\n\n" + input
            assistant_prompt = output
            conversation1 = {'from': 'human', 'value': human_prompt}
            conversation2 = {'from': 'gpt', 'value': assistant_prompt}
            conversations["conversations"].append(conversation1)
            conversations["conversations"].append(conversation2)
            json_obj.append(conversations)
            

current_data.extend(json_obj)

print(len(current_data))

with open('data/updated_chatml_20.json', 'w') as f:
    json.dump(current_data, f, indent=4)
