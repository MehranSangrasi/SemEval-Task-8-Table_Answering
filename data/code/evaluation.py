import pandas as pd
import ast  # For safely evaluating string representations of lists

predictions = []
answers = []
score = 0

df = pd.read_csv('data/filtered_dataset.csv')

# df['answers'] = df['answers'].apply(ast.literal_eval)

# print(df['answers'][21])

# df['answers'] = ast.literal_eval(df['answers'])

predictions.append(df['answers'].tolist())
answers.append(df['actual_answer'].tolist())


for i in range(len(answers[0])):
    prediction = predictions[0][i]
        # import pdb; pdb.set_trace()
    answer = answers[0][i]
    for j in range(len(answers[0][i])):
        if answer.lower()=='true':
            answer = True
            
            
            # print(answer)
            
        elif answer.lower() == 'false':
            answer = False
            
            # print(answer)
            
        if prediction.lower() == 'true':
            prediction = True
            break
            # print(prediction)
            
        elif prediction.lower() == 'false':
            prediction = False
            break
            # print(prediction)
            
        if answer.startswith("[") and answer.endswith("]"):
            answer = ast.literal_eval(answer)
            # print(answer)
            
        if prediction.startswith("[") and prediction.endswith("]"):
            prediction = ast.literal_eval(prediction)
            break
            # print(prediction)
            
        if answer[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            import pdb; pdb.set_trace()
            if "." in answer:
                answer = float(answer)
            else:
                answer = int(answer)
                
            
        if prediction[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            import pdb; pdb.set_trace()
            if "." in prediction:
                prediction = float(prediction)
                
            else:
                prediction = int(prediction)
            
            break

            # print(answer)

            
    if answer == prediction:  
        score += 1
        
            
            
        
        
            
        
            
        # elif value==lowercase():
        #     value = False
        #     print(value)
        
        # if (value.startswith("[") and value.endswith("]")):
        #     # import pdb; pdb.set_trace()
        #     value = ast.literal_eval(value)
        #     # print(type(value))
            # print(value)
        
        # elif isinstance(value, float):
        #     value = float(value)
        #     print(type(value))
        
        # elif isinstance(value, int):
        #     value = ast.literal_eval(value)
            
            
            
        
        # if (predictions[0][i])
        
        # if (type(ast.literal_eval(answers[0][i])) in [list, float, int]):
            
        #     actual = ast.literal_eval(answers[0][i])
        #     evaluated = ast.literal_eval(predictions[0][i])
            
        # else:
        #     actual = answers[0][i]
        #     evaluated = predictions[0][i]
            
        
        # if evaluated == actual:
        #     score += 1
        
        # print(f'Index {i} | Before Preprocessing: {answers[0][i]}')
        # print(f'Index {i} | After Preprocessing: {preprocess(answers[0][i], i)}')
    # pred = preprocess(predictions[0][i], i)
    # ans = preprocess(answers[0][i], i)
    
    # if pred == ans:
    #     score += 1

accuracy = score / len(answers[0])
print(f'Accuracy : {accuracy * 100:.2f}%')
