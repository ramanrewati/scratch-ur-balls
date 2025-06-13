import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X , y = datasets.make_regression(n_samples=1000,n_features=1,noise=30,random_state=10)

X_train , X_test , y_train , y_test = train_test_split( X , y , test_size=0.2 , random_state=169)

plt.figure(figsize=(8,6))
plt.scatter( X[:,0] , y , color='b' , marker = 'o' , s=30 )
plt.show()

linReg = LinearRegression(lr=0.03,n_iters=1000) 
linReg.fit(X_train,y_train)
predictions = linReg.predict(X_test)
print(predictions)

def mse(y_test,predictions):
    return np.mean(y_test - predictions)**2

mse(y_test,predictions)

y_pred_line = linReg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train,y_train,color=cmap(0.9),s=10)
m2 = plt.scatter(X_test,y_test,color=cmap(0.5),s=10)
plt.plot(X,y_pred_line,color='black',linewidth=2,label='Prediction')
plt.show()