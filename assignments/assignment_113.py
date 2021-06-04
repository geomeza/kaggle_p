import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(os.getcwd() + '/titanic/processed_titanic.csv')

ratings = df['Survived']
cols = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']
df = df[cols]

for col_1 in cols:
    for col_2 in cols:
        if '=' in col_1 and '=' in col_2:
            eq_i_1 = col_1.index('=')
            eq_i_2 = col_2.index('=')
            if col_1[:eq_i_1] != col_2[:eq_i_2] and cols.index(col_1) < cols.index(col_2):
                df[col_1 + ' * ' + col_2] = np.array([df[col_1][i]*df[col_2][i] for i in range(len(df[col_1]))])
        elif ('SibSp' in col_1 and 'SibSp' in col_2):
            continue
        else:
            if cols.index(col_1) < cols.index(col_2):
                df[col_1 + ' * ' + col_2] = np.array([df[col_1][i]*df[col_2][i] for i in range(len(df[col_1]))])

cols = [x for x in df.columns]

good_score = 0
interim = None
added_columns = []

for index in range(len(cols)):
    if len(added_columns) == 0:
        added_columns = [cols[0]]
    else:
        added_columns.append(cols[index])
    new_df = df[added_columns]
    training_x = np.array(new_df[:500])
    training_y = np.array(ratings[:500])
    testing_x = np.array(new_df[501:])
    testing_y = np.array(ratings[501:])
    coeffs = LogisticRegression(max_iter=1000).fit(training_x, training_y)
    interim = coeffs.score(testing_x, testing_y)
    if interim > good_score:
        good_score = interim
    else:
        del added_columns[-1]

print("\tTraining:", coeffs.score(training_x, training_y))
print("\tTesting:", coeffs.score(testing_x, testing_y))