import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries (1).csv')
X=dataset.iloc[:,1:-1]
Y=dataset.iloc[:,-1]

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

x_grid= np.arange(0,10,0.01)
x_grid= x_grid.reshape(len(x_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(x_grid,regressor.predict(x_grid))
plt.title('DECISION TREE REG')
plt.show()

