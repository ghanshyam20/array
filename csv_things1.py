# Working with CSv files


# steps to read and display csv file 


# 1 import pandas: first make sure you have pandas library installed in your python env




#  Read the csv files : use the pandas.read_csv() function to load the csv file into a dataframe



# Display the data: Use the head() and tail() methods to previews the data



import pandas as pd


df=pd.read_csv('exams.csv')
print("first five rows of the data")


print(df.head())



print("last five rows of the data")


print(df.tail())


