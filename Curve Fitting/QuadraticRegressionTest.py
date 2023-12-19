import math as m
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



x_data = pd.Series([0.1 , 1 , 1.8 , 3.4 , 4.5])

y_data = pd.Series([1.2 , 1.9 , 2.3 , 1.9 , 1.5])

z_data = pd.Series([4 , 12 , 5 , 2 , 7])


coords = pd.DataFrame({"x_data":x_data,"y_data":y_data,"z_data":z_data})
print(coords)

#Defining quadratic regression model

def model_f(x,a,b,c):
    return a*x**2+b*x+c

#_____________________________________________________________________________________________________________    
# def model_f(x,a,b,c):
#     return a*(x-b)**2+c


# popt, pcov = curve_fit(model_f,x_data,y_data,p0=[1,2,3]) #p0 is three initial guesses to the data points

# a_opt, b_opt, c_opt = popt

# x_model = np.linspace(min(x_data),max(x_data),100)
# y_model = model_f(x_model,a_opt,b_opt,c_opt)

# plt.scatter(x_data,y_data)
# plt.plot(x_model,y_model,color = 'r')
# plt.show()
#_____________________________________________________________________________________________________________

#Quadratic Regression

np_curve = np.polyfit(x_data,y_data,2)

a_opt, b_opt, c_opt = np_curve

print(np_curve)

x_model = np.linspace(min(x_data),max(x_data),100)
y_model = model_f(x_model,a_opt,b_opt,c_opt)
plt.plot(x_model,y_model, color = 'r')
plt.scatter(x_data,y_data)
plt.show()

