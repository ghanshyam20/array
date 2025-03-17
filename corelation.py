# as an example let us compute the correaltion  coefficieent for student height and weight using pandas and our data set




import pandas as pd


df=pd.read_csv('weight-height (2).csv')


print(df.corr())