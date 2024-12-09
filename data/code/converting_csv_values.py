import ast
import pandas as pd


predictions = []
answers = []
score = 0


df = pd.read_csv('data/filtered_dataset.csv')

# df['answers'] = df['answers'].apply(ast.literal_eval)

# print(df['answers'][21])

# df['answers'] = ast.literal_eval(df['answers'])

predictions.append(df['answers'].tolist())
answers.append(df['actual_answer'].tolist())

def convert_to_value(val):
    """
    Convert a string to a Python value (boolean, list, int, or float).
    """
    # Convert boolean strings to actual booleans
    import pdb; pdb.set_trace()
    try:
        if val.lower() == 'true':
            return True
        elif val.lower() == 'false':
            return False
    except:
        pass

    try:# If the value starts and ends with brackets, try to convert it to a list
        if val.startswith("[") and val.endswith("]"):
            val = ast.literal_eval(val)
            val = sorted(val)
            return val
    except:
        pass
    # Try to convert to float or int
    try:
        if val.replace(".", "", 1).isdigit() or val[0] == '-' and val[1:].replace(".", "", 1).isdigit():
            # Check for float or integer
            try:
                if "." in val:
                    val = float(val)
                    val = round(val, 2)
                    # import pdb; pdb.set_trace()
                else:
                    return int(val)
            except ValueError:
                pass  # If the value can't be converted, return the original string
    except:
        pass

    return val  # Return the original value if it doesn't match any of the above cases

def evaluate_predictions(answers, predictions):
    score = 0
    for i in range(len(answers[0])):
        answer = answers[0][218]
        prediction = predictions[0][218]

        # Convert both answer and prediction to appropriate values
        answer = convert_to_value(answer)
        prediction = convert_to_value(prediction)

        if answer == prediction:
            score += 1

    return score


score = evaluate_predictions(answers, predictions)
accuracy = score / len(answers[0])
print(accuracy)
