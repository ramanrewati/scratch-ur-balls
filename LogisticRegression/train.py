import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

bc = datasets.load_breast_cancer()
X , y = bc.data , bc.target
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=6969)

model = LogisticRegression(lr=0.01,n_iters=10000)
model.fit(X_train , y_train)
predictions = model.predict(X_test)

def accuracy(y_pred , y_test):
    acc = np.sum(y_pred == y_test)/len(y_test)
    return acc

acc = accuracy(predictions,y_test)
print(acc)