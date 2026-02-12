
import numpy as np
import pandas as pd

df = pd.read_csv("covid_toy - covid_toy.csv")
# print(df.head(2))

df = df.dropna()
# print(df.dropna())
x = df.drop(columns=['has_covid'])
y = df['has_covid']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.2, random_state=42)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

df['gender'] = lb.fit_transform(df['gender'])
df['cough'] = lb.fit_transform(df['cough'])
df['city'] = lb.fit_transform(df['city'])

print(df.head(3))