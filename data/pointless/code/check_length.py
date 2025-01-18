import json

with open("data/updated_queries_transformation.json") as f:
    data = json.load(f)
    
    
print(len(data))