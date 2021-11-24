import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from Classification.GNBayes import GaussianNaiveBayes
from Regression.LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split

from Regression.LogisticRegression import LogisticRegression
from Regression.utils import accuracy

gnb = GaussianNaiveBayes()

X , y = make_classification(n_classes=2 , n_features=8 , n_samples=1000 )

x_train , x_test , y_train , y_test = train_test_split(X , y ,test_size=0.3)

gnb.fit(x_train , y_train)

y_pred = gnb.predict(x_test)

print(accuracy(y_pred,y_test))