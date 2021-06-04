import numpy as np
import pandas as pd
from sklearn.neighbors import  KNeighborsClassifier
import os

df = pd.read_csv(os.getcwd() + '/wacky_final/flowers.csv')
print(df.columns)
cols = [x for x in df.columns if x not in ['Id', 'Species']]
ratings = df['Species']
new_df = df[cols]
for col in cols:
    print(col, df[col].mean())
print(cols)

unique_species = df['Species'].unique()
for species in unique_species:
    print(species)
    print('-----------------------------------')
    for col in cols:
        print(col, df[col].mean())
    print('-----------------------------------')


for col in cols:
    max = df[col].max()
    minim = df[col].min()
    df[col].apply(lambda x: (x-minim)/(max-minim))
    print('aplied')

df = df.sample(frac=1).reset_index(drop=True)

mid_point = len(ratings)//2
training_x = np.array(new_df[:mid_point])
training_y = np.array(ratings[:mid_point])
testing_x = np.array(new_df[mid_point:])
testing_y = np.array(ratings[mid_point:])

best_accuracy = 0
best_k_val = 0

for i in range(50):
    knn = KNeighborsClassifier(n_neighbors = i+1)
    knn.fit(training_x, training_y)
    accuracy = knn.score(testing_x, testing_y)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k_val = i+1

print(best_accuracy, best_k_val)