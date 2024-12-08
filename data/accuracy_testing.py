import pandas as pd
import os
import numpy as np

predictions = []
answers = []
score = 0

data = pd.read_csv('./data/queries_with_answers.csv')

predictions.append(data['answers'].tolist())
answers.append(data['actual_answer'].tolist())

for i in range(len(answers[0])):
    # import pdb; pdb.set_trace()
    if predictions[0][i] == answers[0][i]: 
        score += 1 

accuracy = score / len(answers[0])
print(f'Accuracy : {accuracy*100}')
