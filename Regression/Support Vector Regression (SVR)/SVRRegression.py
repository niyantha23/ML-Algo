import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values

#feature scaling
Y=Y.reshape(len(Y),1)
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)
sc_Y=StandardScaler()
Y=sc_Y.fit_transform(Y)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)

predict=sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

plt.scatter(sc_X.inverse_transform(X),sc_Y.inverse_transform(Y),color='red')
plt.plot(sc_X.inverse_transform(X),sc_Y.inverse_transform(regressor.predict(X)))
plt.title('SVR REGRESSION')
plt.xlabel('job level')
plt.ylabel('salary')
plt.show