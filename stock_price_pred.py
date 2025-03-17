# stock price prediction using  we will predict the price of gold



import numpy as np

import pandas as pd

import yfinance as yf


import seaborn as sns

import matplotlib.pyplot as plt

# %matplotlib inline




stocks=input("enter the code of the stock ")

data=yf.download(stocks,'2008-01-01','2021-01-01',auto_adjust=True)


data.head()
data.shape


