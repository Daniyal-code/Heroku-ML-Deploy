# Simple Linear Regression

# Importing the libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=1/3, random_state=0)

# Training the Simple Linear Regression model on the Training set
# Since we have a very small dataset, we will train our model with all availabe data.
regressor = LinearRegression()
regressor.fit(X, y)


# Saving model to disk
# this model.pkl file i will deploy to heroku(paas)
pickle.dump(regressor, open('model.pkl', 'wb'))


# Loading model to compare the results

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[4.3]]))