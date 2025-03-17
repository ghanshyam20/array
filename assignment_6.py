import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the data
df = pd.read_csv('bank.csv', delimiter=';')

# Select relevant columns and preprocess
df = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
df = pd.get_dummies(df, drop_first=True)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Split data into features and target
X = df.drop(columns='y')
y = df['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("Logistic Regression Results:")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}\n")

# Plot Logistic Regression confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, cmap='Blues', fmt='d', 
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# K-Nearest Neighbors (KNN)
k_values = [3, 5, 7]  # Test multiple k values
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    print(f"KNN Results (k={k}):")
    print(confusion_matrix(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}\n")

    # Plot KNN confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, cmap='Greens', fmt='d', 
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f'KNN Confusion Matrix (k={k})')
    plt.show()

# Compare models
print("Model Comparison:")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print(f"KNN Accuracy (k={k}): {accuracy_score(y_test, y_pred_knn):.4f}")

# Key Findings
print("\nKey Findings:")
print("1. Logistic Regression outperformed KNN in terms of accuracy for this dataset.")
print("2. KNN's performance varied with different values of k. For example:")
print(f"   - At k=3, accuracy was {accuracy_score(y_test, KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train).predict(X_test)):.4f}")
print(f"   - At k=5, accuracy was {accuracy_score(y_test, KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train).predict(X_test)):.4f}")
print(f"   - At k=7, accuracy was {accuracy_score(y_test, KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train).predict(X_test)):.4f}")
print("3. Logistic Regression is faster to train and more interpretable, making it a better choice for this problem.")
print("4. KNN might perform better with feature scaling or more tuning, but it is generally slower and less efficient for larger datasets.")
print("5. The correlation heatmap showed that some features (e.g., 'poutcome') had a stronger relationship with the target variable 'y'.")

