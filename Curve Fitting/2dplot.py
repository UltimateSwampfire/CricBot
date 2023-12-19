from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import time

######### PREDEFINED TESTING POINTS OF BALL ######################

def projectilepath(a,b,c,x):
    return a*x*x+b*x+c

x = np.linspace(1,40,40)

y = x*(40-x)
y = y+(2*np.random.rand(len(y))-1)*20

###############################


x_plot = np.array([])
y_plot = np.array([])

for i in range(0,len(x)):

        x_plot = np.append(x_plot,x[i])
        y_plot = np.append(y_plot,y[i])

        ########## ALGORITHM #################

        a , b , c  = np.polyfit(x_plot,y_plot,2)

        ########### PLOTTING 3D CURVE ##########

        x_curve = np.linspace(min(x_plot),40,100)
        y_curve = projectilepath(a,b,c,x_curve)


        plt.xlim([0,max(x)])
        plt.ylim([0,max(y)])


        plt.plot(x_curve,y_curve)
        plt.scatter(x[0:i+1],y[0:i+1])
        plt.draw()
        plt.pause(0.01)
        plt.cla()
