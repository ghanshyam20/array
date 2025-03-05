import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the dataset
data = pd.read_csv('data_banknote_authentication.csv')

# Separate features (X) and target (y)
X = data.drop(columns=['class'])
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Extract values from the confusion matrix
TN, FP, FN, TP = conf_matrix.ravel()

# Calculate the total number of samples
total_samples = TN + FP + FN + TP

# Calculate the proportions of each class
p_class_0 = (TN + FP) / total_samples  # Proportion of class 0
p_class_1 = (TP + FN) / total_samples  # Proportion of class 1

# Calculate the Gini index
gini_index = 1 - (p_class_0**2 + p_class_1**2)
print(f"Gini Index: {gini_index:.4f}")

