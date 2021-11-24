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