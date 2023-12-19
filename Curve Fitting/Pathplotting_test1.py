import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits import mplot3d




#Gathering set of points

rx = 10
ry = 10
rz = 10
r0 = [rx,ry,rz] #First detected point coordinates with respect to camera

#All other points must be taken with respect to r0 before being entered in to this program

#<<<CODE TO INPUT COORDS WRT KIENCT CAMERA>>>


#Coords wrt to r0
x_data = pd.Series(
    [    1.7812,
    0.0452,
    1.2793,
    0.6154,
    3.0141

])
y_data = pd.Series([5.3436,
    0.1355,
    3.8379,
    1.8463,
    9.0423




])
z_data = pd.Series([
    6.8100,
    5.3040,
   24.6052,
   17.3066,
    6.235,
])

coords = pd.DataFrame({"X-coordinates" : x_data,"Y-coordinates" : y_data,"Z-coordinates" : z_data})


print(coords)


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

#So far the model only accounts for y=mx throws, does not account for c value.

m1 = (-(x_sum - y_sum) + m.sqrt((x_sum-y_sum)**2+4*(xy_sum)**2))/(2*xy_sum)



#plotting the best fit curve

x_model = np.linspace(min(x_data),max(x_data),5)
y_model = linear_model(x_model, m1, 0)

plt.plot(x_model,y_model,color = 'r')
plt.scatter(x_data,y_data)
plt.show()

#Shifting the x_data points onto the curve

for i in range (0,x_data.size):
    xi = x_data.iloc[i]
    yi = y_data.iloc[i]

    h = (xi/m1 + yi)/(m1+1/m1)
    k = m1*h
    x_data.iloc[i] = h
    y_data.iloc[i] = k

#Updating coords
coords = pd.DataFrame({"X-coordinates" : x_data,"Y-coordinates" : y_data,"Z-coordinates" : z_data})
print(coords)


plt.plot(x_model,y_model, color = 'g')
plt.scatter(x_data,y_data)
plt.show()


#r coordinates

r_data = pd.Series([])

for i in range (0,x_data.size):
    xi = x_data.iloc[i]
    yi = y_data.iloc[i]
    ri = m.sqrt(xi**2+yi**2)
    r_data.loc[i] = ri

print(r_data)

#Now we have the r and z data ready for quadratic regression

def quadratic_model (x,a,b,c):
    return a*x*x+b*x+c

quadratic_curve = np.polyfit(r_data,z_data,2) #Gives best fit values for a b c

a_opt , b_opt , c_opt = quadratic_curve

r_model_quad = np.linspace (min(r_data), max(r_data), 100)
z_model_quad = quadratic_model(r_model_quad,a_opt,b_opt,c_opt)

plt.plot(r_model_quad,z_model_quad,color = 'b')
plt.scatter(r_data,z_data)
plt.show()



#Suppose the bot is in the x-z plane at a distance y0
#Camera coordinates is -r0 = (-rx,-ry,-rz)


cam_coordinates = [-rx,-ry,-rz]

y0 = -ry
#finding other coordinates at bot-plane

x0 = y0/m1
r0 = m.sqrt(x0**2 + y0**2)
z0 = quadratic_model(r0,a_opt,b_opt,c_opt)

print ("FINAL COORDINATES OF BALL : [ ",round(x0+rx,2)," m ,",round(y0+ry,2)," m ,",round(z0+rz,2)," m ]")




