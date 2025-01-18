import pandas as pd
import os
import numpy as np

predictions = []
answers = []
score = 0

df = pd.read_csv('data/filtered_dataset.csv')

# predictions.append(actual_answer['actual_answer'].tolist())
# answers.append(predicted_answer['answers'].tolist())

predictions.append(df['answers'].tolist())
answers.append(df['actual_answer'].tolist())

print(len(answers[0]))

for i in range(len(answers[0])):
    # import pdb; pdb.set_trace()
    if predictions[0][i] == answers[0][i]: 
        score += 1 

accuracy = score / len(answers[0])
print(f'Accuracy : {accuracy*100}')
