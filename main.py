import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# Reading the data
dataset = pd.read_csv("FuelConsumption.csv")
required_dataset = dataset[['ENGINESIZE', 'CO2EMISSIONS']]

# Creating train and test dataset
msk = np.random.rand(len(required_dataset)) < 0.8
train = required_dataset[msk]
test = required_dataset[~msk]

# Training the model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# Making Predictions
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_pred = regr.predict(test_x)

# Making New Predictions
engine_size = 6
emission = regr.predict([[engine_size]])
print(emission[0][0])