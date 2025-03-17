


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Load data
df = pd.read_csv("Assignment 7/data_banknote_authentication.csv")

# Features and target
X = df.drop(columns=['class'])
y = df['class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Train linear SVM
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

# Evaluate linear SVM
y_pred_linear = svm_linear.predict(X_test)
print("Linear Kernel Results:")
print(confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# Train RBF SVM
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

# Evaluate RBF SVM
y_pred_rbf = svm_rbf.predict(X_test)
print("RBF Kernel Results:")
print(confusion_matrix(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# Quick comparison

print("Linear: Best for simple, straight-line separation.")
print(" RBF: Handles complex, non-linear patterns.")


