import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D



def linear_model(x,m):
    return m*x


def quadratic_model(x,a,b,c):
    return a*x*x+b*x+c


################ GENERATE SAMPLE DATASET ################

   
# x = np.array([2,1.10,1.29,-0.1,-1.96,-3])
# y = np.array([10,8.2,8.58,5.78,2.07793,0])
# z = np.array([2,7.19,6.25,11.6992,12.8816,10.541]

#-2.8530   -2.7975   -2.7858   -2.5787   -2.0014   -1.9610   -0.4269   -0.1088    1.2119    1.2906
#0.2941    0.4051    0.4283    0.8426    1.9972    2.0779    5.1462    5.7824    8.4239    8.5811
#11.0034   11.1667   11.2001   11.7504   12.8310   12.8816   12.3903   11.6992    6.6655    6.2557

x = input().split()
x = np.array([float(i) for i in x])
y = input().split()
y = np.array([float(i) for i in y])
z = input().split()
z = np.array([float(i) for i in z])


################# PRINT DATA OF FINAL POINT CONTINUOUSLY ###################


#stores one datapoint at a time to feed into the algorithm

x_queue = [x[0]]
y_queue = [y[0]]
z_queue = [z[0]]
 
x_data = np.array([])
y_data = np.array([])
z_data = np.array([])

i = 0

while True :
    
    #enter current queue data point into out working data
    x_data = np.append(x_data,x_queue[0])
    y_data = np.append(y_data,y_queue[0])
    z_data = np.append(z_data,z_queue[0])   


    if i == 0:

        #defining the initial detected points
        x0 = x_data[i]
        y0 = y_data[i]
        z0 = z_data[i]

        print("Initial set of points : ",[x0,y0,z0])

        # x0 = x[0]
        # y0 = y[0]
        # z0 = z[0]


    i = i+1

    if i==len(x):
        break

    #assign next value in dataset to queue
    x_queue.append(x[i])
    x_queue.pop(0)
    y_queue.append(y[i])
    y_queue.pop(0)
    z_queue.append(z[i])
    z_queue.pop(0)


    if i>1:
            
        ######## USE CURRENTLY AVAILABLE DATA INTO ALGORITHM #########

        #camera coords
        x_cam = -x0
        y_cam = -y0
        z_cam = -z0

        #set every point in reference to first point


        x_rel = np.zeros(x_data.size)
        y_rel = np.zeros(y_data.size)
        z_rel = np.zeros(z_data.size)

        x_rel += x_data - x0
        y_rel += y_data - y0
        z_rel += z_data - z0

        # LINEAR DEMING REGRESSION

        #defining parameters
        
        xs, ys, xys = 0, 0, 0

        for k in range(0,x_rel.size):
            xs+=x_data[k]**2
            ys+=y_data[k]**2
            xys+=x_data[k]*y_data[k]

        m = (-(xs-ys)+np.sqrt((xs-ys)**2+4*(xys**2)))/(2*xys)

        #shifiting datapoints onto curve

        for k in range (0,x_rel.size):
            x_rel[k] = (x_rel[k]+m*y_rel[k])/(m**2+1)
            y_rel[k] = m*x_rel[k]
            

        r_rel = np.sqrt(x_rel**2 + y_rel**2)

        #QUADRATIC REGRESSION

        a , b , c = np.polyfit(r_rel,z_rel,2)

        #FINAL COORDINATES

        #Coordinates when the projected path crosses the camera's x-z plane (y is the depth).
        #Relative to the first point, it is the plane y = y_cam - coords (xf, y_cam, zf)
        #In the camera frame, it is in the plane y = 0, so point (xf-x0,0,zf-z0)

        xf = y_cam/m
        yf = y_cam
        rf = np.sqrt(xf**2+yf**2)
        zf = a*(rf**2)+b*rf+c

        final_coords = np.array([xf+x0,yf+y0,zf+z0])

        print("Ball is approaching : ",final_coords)

        #PLOTTING the changing graphs



print(x_data)
print(y_data)
print(z_data)


fig = plt.figure()
ax = plt.axes(projection='3d')

y_line_rel = np.linspace(0,max(y_data),100)
x_line_rel = y_line_rel/m

r_line_rel = np.sqrt(x_line_rel**2+y_line_rel**2)
z_line_rel = a*(r_line_rel**2)+b*r_line_rel+c

y_line = y_line_rel - y_cam
x_line = x_line_rel - x_cam
z_line = z_line_rel - z_cam 

ax.plot3D(x_line,y_line,z_line)
ax.scatter3D(x_data,y_data,z_data)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.show()











    