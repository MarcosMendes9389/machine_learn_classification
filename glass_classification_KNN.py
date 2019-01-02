from __future__ import print_function
import pandas
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# Load dataset
url = "glass_data/glass.data"
names = ['id', 'refractive index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron', 'Type']
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)
# head
print(dataset.head(20))
# deions
print(dataset.describe())
# class distribution
print(dataset.groupby('Type').size())

# Split-out validation dataset
array = dataset.values
X = array[:, 1:10]
Y = array[:, 10]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
scoring = 'accuracy'

kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(KNeighborsClassifier(n_neighbors=1), X_train, Y_train, cv=kfold, scoring=scoring)
msg = "Result %s: %f (%f)" % ('KNN', cv_results.mean(), cv_results.std())
print(msg)


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))














