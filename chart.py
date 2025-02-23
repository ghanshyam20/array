#data to plot

import matplotlib.pyplot as plt
labesl=['python','c++','ruby','java']
sizes=[215,130,245,210]
colors=['gold','yellowgreen','lightcoral','lightskyblue']
explode=(0.1,0,0,0)
plt.pie(sizes,explode=explode,labels=labesl,colors=colors,autopct='%1.1f%%',shadow=True)
plt.axis('equal')
plt.show()

