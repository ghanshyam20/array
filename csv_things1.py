# # Working with CSv files


# # steps to read and display csv file 


# # 1 import pandas: first make sure you have pandas library installed in your python env




# #  Read the csv files : use the pandas.read_csv() function to load the csv file into a dataframe



# # Display the data: Use the head() and tail() methods to previews the data



# import pandas as pd


# df=pd.read_csv('exams.csv')
# print("first five rows of the data")


# print(df.head())



# print("last five rows of the data")


# print(df.tail())



import pandas as pd


df=pd.read_csv('iris.csv')

print(df.head())

print(df.tail())



print(df.describe())




print(df.dtypes)

print(df.index)


print(df.columns)


print(df.values)



df2=df.sort_values('sepal_length',ascending=False)

print(df2.head())

#pandaa provides powerful methods to slice data frames  we start with slicing columns by name



print(df[['sepal_width']]
      )
print(df[['sepal_width','sepal_length']])



# slicing by rows is done as follows



print(df[2:4])


# slicing by rows and columns at the same time is done uses the function loc() and iloc() functions


print(df.loc[2:4,['petal_width','petal_length']])
print(df.iloc[2:4,[0,1]])


# what  is cost function ==???  

# cost function is a function that measures the performance of a model for given data. It quantiifies the error between prediction 



# cost function what we do is to minimize the cost function to get the best model for our data 

# lets formulate the cost function for linear regression model


# cost function =j(theta)


# then what is j(theta)???

# j(theta) is the cost function for linear regression model. it is the sum of the squared differences between the prediction 


 


# 



# primer on lines and parabolas 




# These points can be plotted in a coordinate system to reveal the graph of the line it should be noted here that one can use any value for x poitive or negative and get the value for y 


import numpy as np

import matplotlib.pyplot as plt

x=np.array([0,1,2,3,4])

y=np.array([1,3,5,7,9])


plt.scatter(x,y)


plt.plot(x,y)


plt.show()

