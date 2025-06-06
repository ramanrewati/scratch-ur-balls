import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF6F61','#6B5B95','#88B04B'])
from KNN import KNN

iris = datasets.load_iris()
X,y = iris.data , iris.target

X_train , X_test , y_train  , y_test = train_test_split(X, y, test_size=0.2 , random_state=69)

plt.figure()
plt.scatter(X[:,2],X[:,3],c=y,cmap=cmap,edgecolor='k',s=20)
plt.show()

clf = KNN(5)

clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print(predictions)

accuracy = np.sum(predictions == y_test) / len(y_test)

print(f"The accuracxy of this KNN model is {accuracy}.")