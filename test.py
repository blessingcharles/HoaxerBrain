import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from Regression.utils import accuracy
from clustering.Kmeans import KMeans
from ensemble.Adaboost import Adaboost
from ensemble.RandomForest import RandomForest
import pandas as pd


cancer = datasets.load_breast_cancer()

X , y = cancer.data , cancer.target

x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.3) 
km = KMeans()


km.predict(X)
km.plot()