import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

dataframe = pd.read_csv(os.getcwd() + '/titanic/processed_titanic.csv')

ratings = dataframe['Survived']
dataframe = dataframe[['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']]
training_x = np.array(dataframe[:500])
training_y = np.array(ratings[:500])
testing_x = np.array(dataframe[501:])
testing_y = np.array( ratings[501:])

coeffs = LogisticRegression().fit(training_x, training_y)
print("Constant", coeffs.intercept_[0])
final_coeffs = {column: coeff for column, coeff in zip(dataframe.columns, *coeffs.coef_)}
for col,value in final_coeffs.items():
    print(col,value)
print("\n")

print("training:", coeffs.score(training_x, training_y))
print("testing:", coeffs.score(testing_x, testing_y))