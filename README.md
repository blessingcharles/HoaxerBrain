"""  
# LinearRegression

df = pd.read_csv("datasets/houseprice.csv")
sx = MinMaxScaler()
sy = MinMaxScaler()

scaled_x = sx.fit_transform(df.drop("price",axis=1))
scaled_y = sy.fit_transform(df["price"].values.reshape(-1,1)).reshape(len(df["price"].values),)

x_train , x_test , y_train , y_test = train_test_split(scaled_x , scaled_y , test_size=0.3)

lr = LinearRegression()
lr.fit(scaled_x,scaled_y)

predicted_value = lr.predict(x_test)

"""

"""
# Logistic Regression

data = load_breast_cancer()
X , y = data.data , data.target

lg = LogisticRegression(epochs=2000,lr=0.1)
sx = MinMaxScaler()

scaled_x = sx.fit_transform(X)
x_train , x_test , y_train , y_test = train_test_split(scaled_x , y , test_size=0.3)

lg.fit(x_train,y_train)

y_pred = lg.predict(x_test)

"""

"""
# Gaussian Naive Bayes

gnb = GaussianNaiveBayes()

X , y = make_classification(n_classes=2 , n_features=8 , n_samples=1000 )

x_train , x_test , y_train , y_test = train_test_split(X , y ,test_size=0.3)

gnb.fit(x_train , y_train)

y_pred = gnb.predict(x_test)

print(accuracy(y_pred,y_test))

""""

""""

# PERCEPTRON 


p = Perceptron(epochs=100)

X , y = datasets.make_blobs(n_samples=800 , n_features=2 , centers=2)
print(np.unique(y))
x_train , x_test , y_train , y_test = train_test_split(X , y ,test_size=0.3)

p.fit(x_train,y_train)
y_pred = p.predict(x_test)

print(accuracy(y_test , y_pred))


""""

"""
# DECISION TREE (ID3 ALGORITHM) 

cancer = datasets.load_breast_cancer()

X , y = cancer.data , cancer.target

x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.3)
dt = DecisionTree()

dt.fit(x_train , y_train)

y_pred  = dt.predict(x_test)

print(accuracy(y_test , y_pred))

"""

"""
# PRINCIPAL COMPONENT ANALYSIS

data = load_iris()
X = data.data
y = data.target
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)
print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

"""