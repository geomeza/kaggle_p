import pandas as pd
import numpy as np

df = pd.read_csv('/home/runner/kaggle/2_6/justin_csv.csv')
print(df['training_hours'].mean())

changes = df[df['target'] == 1]
print(len(changes)/ len(df['target']))

print(df['city'].max())

lol = df['company_size'].dropna()

lol = lol.astype(int)

changes = lol[lol < 10]
print(len(changes))

