import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# 1 create sample data (Income,credit,score,loan approved)



data={
    'Income':[50000, 60000, 45000, 80000, 30000, 90000, 20000, 75000, 40000, 65000],
    'credit':[700, 750, 680, 800, 600, 820, 500, 780, 640, 720],
    'loan_approved':[1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  #approved=1, not approved=0

}



df=pd.DataFrame(data)

# 2. Split data into Features (X) and Target (y)
X = df[["income", "credit_score"]]  # Features: Income & Credit Score
y = df["loan_approved"]  # Target: Loan Approval (1 or 0)

# 3. Split into Training & Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Predict on Test Data
y_pred = model.predict(X_test)

# 6. Print Accuracy Score
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. Test with a New Customer
new_customer = [[70000, 750]]  # Example: Income = 70,000, Credit Score = 750
prediction = model.predict(new_customer)

print("Loan Approval Prediction:", "Approved" if prediction[0] == 1 else "Rejected")
