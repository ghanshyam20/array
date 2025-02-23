import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



#sample data (Hours studied vs Exam score)


x=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y=np.array([10,20,30,35,50,65,70,85,95,100])


#creating model and train


model=LinearRegression()
model.fit(x,y)



#make prediction

y_pred=model.predict(x)


#plot results


plt.scatter(x,y,color='blue',label='Actual data')
plt.plot(x,y_pred,color='red',linewidth=2,label='Regression Line')
plt.xlabel('Hours studied')
plt.ylabel('Exam score')
plt.legend()
plt.show()


#print coefficient and intercept


print(f"Intercept: {model.intercept_}")
print(f"slope: {model.coef_[0]}")


