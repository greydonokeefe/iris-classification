from sklearn.datasets import load_iris  # function
from sklearn.model_selection import train_test_split  # function
from sklearn.neighbors import KNeighborsClassifier  # class
import matplotlib.pyplot as plt  # sub module w/ alias
import numpy as np  # module w/ an alias

iris = load_iris()
print(type(iris))
print(iris.keys())  # functions like a dictionary
print(iris)
# print(iris.data)  # full dataset
# print('Target Names:', iris['target_names'])
# print('Feature Names:', iris['feature_names'])
# print('Description:', iris['DESCR'])

# training features, testing features, training labels, testing (obscured) labels
# input: data = X features, target = output/labels,
#       test_size => 25% reserved for testing/validation, random 0 => reproducible
# return: X = features, y = labels/output
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)  # __init__ return an object of knn
# fit method of the knn object trains the model
# training data, training target
# X_train is our feature input
# y_train is our desired output
knn.fit(X_train, y_train)  # testing portion is reserved for validation

# measuring a flower in the wild (unlabeled)
# Feature Names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X_new = np.array([[5.8, 3.1, 5., 1.7]])  # unknown

prediction = knn.predict(X_new)
prediction = int(prediction[0])
print("prediction:", prediction, iris["target_names"][prediction])

y_pred = knn.predict(X_test)
print("y_pred:", y_pred[::-1])
pred_target_names = [iris['target_names'][y_pred[i]] for i in y_pred]
print(f'{pred_target_names[::-1]=}')
print("y_test:", y_test[::-1])
actual_target_names = [iris['target_names'][y_test[i]] for i in y_test]
print(f'{actual_target_names[::-1]=}')

print("mean accuracy:", knn.score(X_test, y_test))



