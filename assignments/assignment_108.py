import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

df = pd.read_csv(os.getcwd() + '/titanic/processed_titanic.csv')

coeffs = ['Survived','Sex','Pclass','Fare','Age','SibSp', 'SibSp>0','Parch>0','Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']
df_train = df[coeffs][:500]
df_test = df[coeffs][500:]



train = np.array(df_train)
test = np.array(df_test)

Y_train = train[:,0]
X_train = train[:,1:]

Y_test = test[:,0]
X_test = test[:,1:]

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

cols = ['Constant']+coeffs[1:]
coefs = [regressor.intercept_]+[x for x in regressor.coef_]

print({cols[i]:round(coefs[i],4) for i in range(len(cols))})

for data in [(X_test,Y_test),(X_train,Y_train)]:
    predictions = regressor.predict(data[0])

    result = [0,0]
    for i in range(len(predictions)):
        output = 1 if predictions[i]>0.5 else 0
        result[1]+=1
        if output == data[1][i]:
            result[0]+=1

    print(result[0]/result[1]) 