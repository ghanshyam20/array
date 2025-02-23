#this is all i want to explore seaborn tutorial


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#data to plot
labels=['python','c++','ruby','java']
sizes=[215,130,245,210]
colors=['gold','yellowgreen','lightcoral','lightskyblue']
explode=(0.1,0,0,0)
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)
plt.axis('equal')
plt.show()
