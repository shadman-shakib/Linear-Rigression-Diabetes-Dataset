from prompt_toolkit import validation
from sklearn import *
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()

X = diabetes.data[:, np.newaxis, 3]
# print(X.shape)

Y = diabetes.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


reg = linear_model.LinearRegression()

reg.fit(X_train, Y_train)

Y_pred = reg.predict(X_test)

Coef = reg.coef_
print(Coef)

R2 = r2_score(Y_test, Y_pred)

mse = mean_squared_error(Y_test, Y_pred)
print(R2, mse)