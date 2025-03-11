# # we get ourselfs familiar with the simple linear regression model



# # this first step is import data from datafile to python using pandas and inspect the data in raw numbers



# import pandas as pd
# import numpy as np

# import matplotlib.pyplot as plt

# data=pd.read_csv('linreg_data(1).csv',skiprows=0,names=['x','y'])
# # head.data

# # it is also a good idea to form visual understanding of the data by plotting it 



# xpd=data['x']

# ypd=data['y']


# n=xpd.size

# plt.scatter(xpd,ypd)

# plt.show()


# # the computation of the means of x and y using numpy is as easy as 


# xbar=np.mean(xpd)

# ybar=np.mean(ypd)


# term1=np.sum(xpd*ypd)

# term2=np.sum(xpd**2)

# b=(term1-n*xbar*ybar)/(term2-n*xbar*xbar)

# a=ybar-b*xbar


# x=np.linspace(0,2,100)

# y=a+b*x



# plt.plot(x,y,color='black')

# plt.scatter(xpd,ypd)


# plt.scatter(xbar,ybar,color='red')


# plt.show()










                       


# using sklearn 

# first thing is to do is import modules from sklearn we import only the linear model




import numpy as np

import matplotlib.pyplot as plt


from sklearn import linear_model


# next we read in data and convert it into 2 D arrya




my_data=np.genfromtxt('linreg_data(1).csv',delimiter=',')
xp=my_data[:,0]
yp=my_data[:,1]

xp=xp.reshape(-1,1)
yp=yp.reshape(-1,1)



# model is created and trained in just two lines of code 


reg=linear_model.LinearRegression()
reg.fit(xp,yp)  #  fitting the model=trainin the model


# the coefficients of the model are a and b are now attributes of regr object


print(reg.coef_,reg.intercept_)


# this gives us the same exact results as our low-level aproach 



# making predictions is done as follows



xval=np.full((1,1),0.5)

yval=reg.predict(xval)

print(yval)



# also this gives the same results as above 



# plotting of the regression line can be done by first predicting y-values for some appropriate x values


xval=np.linspace(-1,2,20).reshape(-1,1)


yval=reg.predict(xval)

plt.plot(xval,yval)   # this plots the line


plt.scatter(xp,yp,color='red') # this plots the data points



plt.show()

# regarding the metrics of accuracy introduced earlier they are readily available in sklearn import metrcis 



yhat=reg.predict(xp)


print('mean absolute erro',metrics.mean_absolute_error(yp,yhat))
print('mean squared erro',metrics.mean_squared_error(yp,yhat))
print('root squared error',np.sqrt(metrics.mean_squared_error(yp,yhat)))

print('r2 value',metrics.r2_score(yp,yhat))











