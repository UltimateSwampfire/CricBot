import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt


#INPUT X AND Y DATA POINTS
x_data = pd.Series([
        
    2.6477,
   10.8600,
    1.9285,
    10.2098,
    6.0601
])


y_data = pd.Series([
    9.3364,
  -56.3624,
    8.6277,
  -39.7454,
   -2.5724
])



coords = pd.DataFrame({"X-coordinates" : x_data,"Y-coordinates" : y_data})

def quadratic_model (x,a,b,c):
    return a*x*x+b*x+c

#GENERATING BEST FIT CURVE
quadratic_curve = np.polyfit(x_data,y_data,2) #Gives best fit values for a b c

#OPTIMAL PARAMETERS
a_opt , b_opt , c_opt = quadratic_curve

#PLOTTING BEST FIT CURVE AND DATA POINTS
x_model_quad = np.linspace (min(x_data), 2*max(x_data), 200)
y_model_quad = quadratic_model(x_model_quad,a_opt,b_opt,c_opt)

plt.plot(x_model_quad,y_model_quad,color = 'b')
plt.scatter(x_data,y_data)
plt.show()



