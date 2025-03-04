import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Load the data from the file
df = pd.read_csv('bank.csv', delimiter=';')

# Step 2: Pick only the important columns we need
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

# Step 3: Change words into numbers so the computer can understand
df2 = pd.get_dummies(df2, drop_first=True)  # This turns categories into numbers
df2['y'] = df2['y'].map({'yes': 1, 'no': 0})  # Change "yes" to 1, "no" to 0

# Step 4: Show a picture (heatmap) to see how numbers relate to each other
sns.heatmap(df2.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Step 5: Pick which columns are the features (X) and which one is the target (y)
X = df2.drop(columns='y')  # Features are everything except 'y'
y = df2['y']  # Target is just 'y'

# Step 6: Split the data into two parts: one for training and one for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Train a model using Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Step 8: Check how good the Logistic Regression model is
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))

# Show confusion matrix as a picture for Logistic Regression
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, cmap='Blues', fmt='d', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Step 9: Train a K-Nearest Neighbors (KNN) model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Step 10: Check how good the KNN model is
print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Show confusion matrix as a picture for KNN
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, cmap='Greens', fmt='d', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('KNN Confusion Matrix')
plt.show()

# Step 11: Compare how well both models did
print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(f"KNN Accuracy (k=3): {accuracy_score(y_test, y_pred_knn):.4f}")

# Findings:
'''
1. Logistic Regression is better at predicting the results in this case.
2. KNN (k=3) is a bit worse because it struggles with more complex data.
3. KNN could work better if we change the number 'k' (the number of neighbors).
4. Logistic Regression is faster and works better for this problem.
'''
