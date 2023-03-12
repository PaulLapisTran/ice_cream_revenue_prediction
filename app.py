import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
import streamlit as st
import pickle
import pandas

df = pandas.read_csv('IceCreamData.csv')
df.head()
# print(df)

X = df['Temperature']
y = df['Revenue']

X_train, X_test, y_train, y_test = tts(X, y, test_size=50)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = np.array(X_train)
X_train = X_train.reshape(-1, 1)
X_test = np.array(X_test)
X_test = X_test.reshape(-1, 1)

model = LinearRegression()
model.fit(X_train, y_train)

filename = 'revenue_prediction.pickle'
pickle.dump(model, open(filename, 'wb'))