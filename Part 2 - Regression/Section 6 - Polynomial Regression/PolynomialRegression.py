import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
LRegressor=LinearRegression()
LRegressor.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,Y)

#visualisation
plt.scatter(X,Y,color='red')
plt.plot(X,LRegressor.predict(X),color='blue')
plt.title('TRUTH OR BLUFF LREG')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_2.predict(X_poly),color='blue')
plt.title('TRUTH OR BLUFF PREG')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

#preditc new result
L_predict=LRegressor.predict([[6.5]])
P_predict=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))