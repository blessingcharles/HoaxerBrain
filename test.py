import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from Regression.LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split

from Regression.LogisticRegression import LogisticRegression
from Regression.utils import accuracy


data = load_breast_cancer()
X , y = data.data , data.target

lg = LogisticRegression(epochs=2000,lr=0.1)
sx = MinMaxScaler()

scaled_x = sx.fit_transform(X)
x_train , x_test , y_train , y_test = train_test_split(scaled_x , y , test_size=0.3)

lg.fit(x_train,y_train)

y_pred = lg.predict(x_test)


print(accuracy(y_test,y_pred))