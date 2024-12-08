import pandas as pd
# import numpy as np
# Load the dataset
df = pd.read_csv("Forbes.csv")

# print(len(df))


# answer =df.loc[(df['Survived'] == 1) & (df['Age'].notnull())].sort_values('Age', ascending=False).head(3)
# Were there any passengers who paid a fare of more than $500?

# fare_bins = [0, 50, 100, 150, float('inf')]
# fare_labels = ['0-50', '50-100', '100-150', '150+']
# df['Fare Range'] = pd.cut(df['Fare'], bins=fare_bins, labels=fare_labels)
exists = df[(df['gender'] == 'Female') & (df['rank'].rank(ascending=False) < 5)]['rank'].values

# For example,  "What's the most common gender among the survivors?":
# ```
# df[df['Survived'] == True]['Sex'].mode()[0]
# ```
# another Example: "Were there any female passengers in the 2nd class who survived?":
# ```
# ((df['Sex'] == 'female') & (df['Pclass'] == 2) & (df['Survived'] == True)).any()
# ```

print(exists)



