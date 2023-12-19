import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m

x_data = pd.Series([1 , 2 , 3 , 4 , 5])

y_data = pd.Series([1.5 , 4.5 , 6.2 , 8.4 , 9.7])

def linear_model(x,m,c):
    return m*x+c

#Defining variables for the deming slope
x_sum, y_sum, xy_sum = [0,0,0]

for i in x_data:
    x_sum+=i**2

for i in y_data:
    y_sum +=i**2

for i in range(0,len(x_data)):
    xy_sum += x_data[i]*y_data[i]


#defining the deming slope

m1 = (-(x_sum - y_sum) + m.sqrt((x_sum-y_sum)**2+4*(xy_sum)**2))/(2*xy_sum)

#m2 = (-(x_sum - y_sum) - m.sqrt((x_sum-y_sum)**2+4*(xy_sum)**2))/(2*xy_sum)

l_curve = np.polyfit(x_data,y_data,1)

print(l_curve)

x_model = np.linspace(min(x_data),max(x_data),100)
y_model = linear_model (x_model, m1, 0)

#plot the scatter points and the best fit curve

plt.plot(x_model,y_model,color = 'r')
plt.scatter(x_data,y_data)
plt.show()




