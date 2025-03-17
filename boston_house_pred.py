import numpy as np


import pandas as pd


import plotly.express as px
import plotly.graph_objects as go

import plotly.io as pio

pio.templates


import seaborn as sns

import matplotlib.pyplot as plt

# %matplotlib inline


from sklearn.datasets import load_boston


load_boston=load_boston()


X=load_boston.data

y=load_boston.target

data=pd.DataFrame(X,columns=load_boston.feature_names)

data['SalePrice']=y

data.head()


# print(load_boston.DESCR)

# print(data.shape)

data.describe()
