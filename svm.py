import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score




# Manually creating 10 data points (time,location,amount)

X=np.array([[10, 1, 100],   # Legit transaction
    [15, 2, 150],   # Legit transaction
    [20, 3, 200],   # Legit transaction
    [25, 1, 3000],  # Fraud transaction
    [30, 2, 3500],  # Fraud transaction
    [35, 3, 4000],  # Fraud transaction
    [40, 1, 500],   # Legit transaction
    [45, 2, 600],   # Legit transaction
    [50, 3, 5500],  # Fraud transaction
    [55, 1, 7000],
      ])  # Fraud transaction




#labels (0=legit, 1=fraud)


y=np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()

# Fit and transform the training data
X_train = scaler.fit_transform(X_train)

# Transform the test data (use the same scaler)
X_test = scaler.transform(X_test)

# Create the SVM model
svm = SVC(kernel='rbf')  # Use RBF kernel

# Train the model
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the results
print("Predictions:", y_pred)
print("Actual Labels:", y_test)
print("Accuracy:", accuracy * 100, "%")