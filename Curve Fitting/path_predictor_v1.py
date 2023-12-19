import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def linear_model (x,m): #Where c = 0 in y=mx+c
    return m*x

def quadratic_model(x, a, b, c):
    return a*x*x+b*x+c

def path_predict(p0,coord):
    

    #Defining variables for the deming slope
    xs, ys, xys = [0,0,0]

    return 0




x0 , y0, z0 = input("Enter first detected point : ").split() #list input [x0, y0, z0]



x0 = float(x0)
y0 = float(y0)
z0 = float(z0)

x_data = np.array([x0])
y_data = np.array([y0])
z_data = np.array([z0])

while True:
    xi , yi, zi = input("Enter next detected point : ").split() #list input [xi, yi, zi]

    xi = float(xi)
    yi = float(yi)
    zi = float(zi)
    
#W A R N I N G  : Repeating np.appends is inefficient and might eat up memory

    x_data = np.append(x_data, xi)
    y_data = np.append(y_data, yi)
    z_data = np.append(z_data, zi)

    #Coordinates with respect to first point coords

    x_rel = x_data - x0
    y_rel = y_data - y0
    z_rel = z_data - z0

    print("X-data is : ",z_rel)
    print("Y-data is : ",y_rel)
    print("Z-data is : ",z_rel)


#We've got the required points, its MATHIN'TIME 

#First up, linear regression
    
    #Defining parameters
    xs, ys, xys= 0, 0, 0

    for i in range (0,x_rel.size):
        xs+=np.square(x_rel[i])
        ys+=np.square(y_rel[i])
        xys+=x_rel[i]*y_rel[i]

    m = (-(xs-ys)+np.sqrt((xs-ys)**2+4*(xys)**2))/(2*xys) #note : 2 m values are there( cus of ±), one which is best fit, other is worst
     #need to make a case arguement to select between them, for now only including one  

    #plotting original linear plot   
    # x_model = np.linspace(min(x_rel),max(x_rel),100)
    # y_model = m*x_model
    # plt.scatter(x_rel,y_rel)
    # plt.plot(x_model,y_model, color = "green")
    # plt.show()

    #Shifting datapoints to linear curve

    for i in range (0,x_rel.size):
        x_rel[i] = (x_rel[i]+m*y_rel[i])/(m**2+1)
        y_rel[i] = m*x_rel[i]


    #plotting shifted linear plot
    # plt.scatter(x_rel,y_rel)
    # plt.plot(x_model,y_model, color = "green")
    # plt.show()

    r_data = np.zeros(x_rel.size,dtype = float)


    r_data += np.sqrt(x_rel**2+y_rel**2)


    #quadriatic regression

    a , b , c = np.polyfit(r_data,z_rel,2)

    # r_model = np.linspace(min(r_data),max(r_data)*2,100)
    # z_model = a*r_model*r_model + b*r_model + c
    # plt.plot(r_model,z_model,color = "red")
    # plt.show()
    
    #PLOTTING THE 3D MODEL

    x_cam = -x0
    y_cam = -y0
    z_cam = -z0

    fig = plt.figure()
    ax = plt.axes(projection = "3d")

    y_line_rel= np.linspace(0,y_cam,100)
    x_line_rel = y_line_rel/m
    rline = np.sqrt(x_line_rel**2+y_line_rel**2)
    z_line_rel = a*(rline**2) + b*rline+c


    xline = x_line_rel - x_cam
    yline = y_line_rel - y_cam
    zline = z_line_rel - z_cam
    ax.plot3D(xline,yline,zline,color = 'blue')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.scatter(x_data+0.1*np.random.randn(x_data.size),y_data+0.1*np.random.randn(y_data.size),z_data+0.1*(np.random.randn(z_data.size)))
    #ax.scatter(x_rel+0.1*np.random.randn(x_rel.size),y_rel+0.1*np.random.randn(y_rel.size),z_rel+0.1*(np.random.randn(z_rel.size)))
    
    X = np.arange(-5,5,0.5)
    Y = np.arange(-5,5,0.5)

    plt.show()
    
    # resp = input ("are ya done?")
    # if resp == "yes":
    #     break
    # elif resp == "no":
    #     continue




