#creating a pandas dataframe



#in the real world a pandas datafram will be created by loading the datasets from existing storage storage can be sql database csv file and excel file pandas datafram can be created from the lists ,dictionary and from a list of dictionary etc

#this is empty frame 
#Data frame can be created jsut by calling a dataframe constructor..



# import pandas as pd

# df=pd.DataFrame()
# print(df)

#creating a datafram using list


import pandas as pd

#list of strings


# lst=['sodium','potassium','calcium','magnesium','iron']


# #calling datafram constructor on list 


# df=pd.DataFrame(lst)

# print(df)



#creating dataframe from dict of ndarray/lists

import pandas as pd


data={'Name':['Ghanshyam','Rajesh','Ramesh','Suresh','Rajesh'],'Age':[20,21,19,18,24]}


df=pd.DataFrame(data)


print(df)


# why use dataframe instead of a dataset ??



#data frame specially designed for data maniulation and analysis offering several advantages over genral datasets,,
# *integrated handeling of missing data
# *powerful data alignment and broadcasting 
# * better performance for operation involving structured data

#*Integrationn with a varietly of data sources and file formats




# what type is a dataframe in pandas??


# import pandas as pd


# df=pd.DataFrame({'A':[1,2,3],'B':[4,5,6]})


# print(type(df))

# <class 'pandas.core.frame.DataFrame'>