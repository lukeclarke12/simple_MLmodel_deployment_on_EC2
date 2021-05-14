#!/usr/bin/env python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris_dict = load_iris()
X = iris_dict['data']
y = iris_dict['target']

from sklearn.utils import shuffle
X_new, y_new = shuffle(X, y, random_state=0)

n_samples_train = 120 # number of samples for training (--> #samples for testing = len(y_new) - 120 = 30)
X_train = X_new[:n_samples_train, :]
y_train = y_new[:n_samples_train]

X_test = X_new[n_samples_train:, :]
y_test = y_new[n_samples_train:]

# model training

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model results
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

# Save the model as a pickle
import pickle

with open('iris_trained_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
