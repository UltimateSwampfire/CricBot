import numpy as np
from collections import deque
import matplotlib.pyplot as plt


def linear_model (x,m): #Where c = 0 in y=mx+c
    return m*x

def quadratic_model(x, a, b, c):
    return a*x*x+b*x+c

#Parameters

no_of_frames = 100  # 60 FPS - 167 ms

v_assumed = 50 #kmph
d_assumed = 10 #meters

pts_lst = [] # list of (no_of_frames) (x,y,z) coords


x_data = np.array([2,1.10,1.29,-0.1,-1.96,-3])
y_data = np.array([10,8.2,8.58,5.78,2.07793,0])
z_data = np.array([2,7.19,6.25,11.6992,12.8816,10.541])


# pts_lst = [(x[i],y[i],z[i]) for i in range(len(x))]


#Algorithm

#Linear regression

#Defining parameters

#Data points wrt camera

# x_data = np.array([tup[0] for tup in pts_lst])
# y_data = np.array([tup[1] for tup in pts_lst])
# z_data = np.array([tup[2] for tup in pts_lst])

#First detected point
x0 = x_data[0]
y0 = y_data[0]
z0 = z_data[0]

#camera coords
x_cam = -x0
y_cam = -y0
z_cam = -z0

#set every point in reference to first point

# x_rel = [x - x0 for x in x_data]
# y_rel = [y - y0 for y in y_data]
# z_rel = [z - z0 for z in z_data]

x_rel = x_data - x0
y_rel = y_data - y0
z_rel = z_data - z0

xs , ys , xys = 0 , 0 , 0

# for i in range(0,len(pts_lst)):
#         xs+=np.square(x_rel[i])
#         ys+=np.square(y_rel[i])
#         xys+=x_rel[i]*y_rel[i]

xs = np.sum(np.square(x_rel))
ys = np.sum(np.square(y_rel))
xys = np.sum(x_rel * y_rel)


m = (-(xs-ys)+np.sqrt((xs-ys)**2+4*(xys)**2))/(2*xys) 


#Shifting datapoints to linear curve

for i in range(0,len(pts_lst)):
        x_rel[i] = (x_rel[i]+m*y_rel[i])/(m**2+1)
        y_rel[i] = m*x_rel[i]


x_rel = (x_rel + m * y_rel)/(m**2 + 1)
y_rel = m * x_rel

r_rel = np.sqrt(x_rel ** 2 + y_rel ** 2)

#QUADRATIC REGRESSION

a , b , c = np.polyfit(r_rel,z_rel,2)


#Final predicted point

#How far ahead we want bat to go:

y_up_front = 0.5 #meters, i.e. get the coordinates when ball will be 0.5m in front of cam


#up_front plane

y_up = y_cam + y_up_front

#Coords in first_point frame

xfp = y_up/m
yfp = y_up
rfp = np.sqrt(xfp**2 + yfp**2)
zfp = quadratic_model(rfp, a, b, c)


#Coords in camera frame

#For camera, y is height and z is depth
#But this calc takes height as z and y as depth
#Before submitting to IK, just interchange y and z coords

xf = round(xfp - x_cam,2) #X 
zf = round(yfp - y_cam,2) #Depth
yf = round(zfp - z_cam,2) #Height

#Final coords
final_coords = (xf,yf,zf)


print("Final_coords are : ({} m, {} m, {} m) ".format(xf,yf,zf))

x_plot = np.arange(min(x_data)*2,max(x_data)*2,0.1)
y_plot= x_plot * m
r_plot = np.sqrt(x_plot**2 + y_plot **2)
z_plot = quadratic_model(r_plot,a,b,c)

x_plot -= x_cam
y_plot -= y_cam
z_plot -= z_cam


fig = plt.figure()

ax = fig.add_subplot(111, projection = '3d')

ax.scatter(x_data,y_data,z_data)
ax.plot(x_plot,y_plot,z_plot)
plt.show()

