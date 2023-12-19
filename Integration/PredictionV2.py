from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


# def parabolic(x, a, b ,c):

#     return a*x*x + b*x + c


# def plane(x,x0,y0,u0,v0):
    
#     t = (x-x0)/u0
#     y = y0 - v0*t
#     return y

# def profile(y,y0,z0,v0,w0):

#     t = (y0 - y)/v0
#     z = z0 + w0*t - 0.5 * 9.81 * t * t
#     return z



# x_data = np.array([2,1.10,1.29,-0.1,-1.96,-3])
# y_data = np.array([10,8.2,8.58,5.78,2.07793,0])
# z_data = np.array([2,7.19,6.25,11.6992,12.8816,10.541])

# #Calculating parameters

# constants_1 = curve_fit(plane,x_data, y_data)
# x1_fit = constants_1[0][0]
# y1_fit = constants_1[0][1]
# u1_fit = constants_1[0][2]
# v1_fit = constants_1[0][3]
# # print("Constants 1 : ",constants_1)

# constants2 = curve_fit(profile,y_data,z_data)
# y2_fit = constants2[0][0]
# z2_fit = constants2[0][1]
# v2_fit = constants2[0][2]
# w2_fit = constants2[0][3]


# y_parametric = np.linspace(0,20,100)

# t = (y1_fit - y_parametric)/v1_fit

# x_parametric = x1_fit + u1_fit * t

# t = (y2_fit - y_parametric)/v2_fit
# z_parametric = z2_fit + w2_fit * t - 0.5*9.81 * t * t



# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.view_init(0,2)
# plt.xlim([-2,2])
# plt.ylim([0,20])

# ax.scatter3D(x_data, y_data, z_data, 'greens')
# ax.plot(x_parametric,y_parametric,z_parametric)

# plt.show()


def parabolic(x, a, b ,c):

    return a*x*x + b*x + c


def plane (x,x0,z0,vx,vz):
    t = (x-x0)/vx
    z = z0 + vz*t
    return z

def profile(z,z0,y0,vz,vy):

    t = (z - z0)/vz
    y = y0 + vy * t - 0.5 * 9.81 * t * t
    return y



x_data = np.array([2,1.10,1.29,-0.1,-1.96,-3])
z_data = np.array([10,8.2,8.58,5.78,2.07793,0])
y_data = np.array([2,7.19,6.25,11.6992,12.8816,10.541])

#Calculating parameters

constants_1 = curve_fit(plane,x_data, z_data)
x1_fit = constants_1[0][0]
z1_fit = constants_1[0][1]
vx1_fit = constants_1[0][2]
vz1_fit = constants_1[0][3]
# print("Constants 1 : ",constants_1)

constants2 = curve_fit(profile,z_data,y_data)
z2_fit = constants2[0][0]
y2_fit = constants2[0][1]
vz2_fit = constants2[0][2]
vy2_fit = constants2[0][3]

x_parametric = np.linspace(-3,3)

t = (x_parametric - x1_fit) / vx1_fit

z_parametric = z1_fit + vz1_fit * t

t = (z_parametric - z2_fit) / vz2_fit

y_parametric = y2_fit + vy2_fit * t - 0.5 * 9.81 * t * t


fig = plt.figure()
ax = plt.axes(projection = '3d')
# ax.view_init(0,2)
# plt.xlim([-2,2])
# plt.ylim([0,20])

ax.scatter3D(x_data, z_data, y_data, 'greens')
ax.plot(x_parametric,z_parametric,y_parametric)
ax.set_xlabel("Width")
ax.set_ylabel("Depth")
ax.set_zlabel("Height")

plt.show()



# fig, ax = plt.subplots(1,2)

# ax[0].scatter(x_data,z_data)
# ax[0].plot(x_parametric,z_parametric)
# ax[0].set_xlim(min(x_data),max(x_data))
# plt.grid(True)


# ax[1].scatter(z_data,y_data)
# ax[1].plot(z_parametric,y_parametric)
# ax[1].set_xlim(min(z_data),max(z_data))


# plt.show()

