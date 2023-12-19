#!/usr/bin/env python3
# Description:
# - Subscribes to real-time streaming video from your built-in webcam.
#
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
#
# Edited by:
# - Shreyash Patidar
# - ME18B074
 
# Import the necessary libraries
import rospy # Python library for ROS
from std_msgs.msg import String
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

samples = np.array([[]])

plt.ion() # turn interactive mode on
fig = plt.figure()
#fig, (ax1, ax2) = plt.subplots(2, projection='3d')
ax = plt.axes(projection='3d')
ax.view_init(0, 2)
plt.xlim([-2, 2])
plt.ylim([0, 20])

def update_figure():
	global samples
	
	# clear canvas
	#ax.cla()
	#fig.clf()
	
	# plot updated data
	xdata = samples[ :,0]
	ydata = samples[ :,1]
	zdata = samples[ :,2]
	
	ax.scatter3D(xdata, ydata, zdata, 'greens')
 
	if (samples.shape[0]%5==0 and samples.shape[0]!=0):
		rospy.loginfo('Finally called \n'*10)	
		
		# curve fitting
		constants_1 = curve_fit(plane,xdataset,ydataset)
		x1_fit = constants_1[0][0]
		y1_fit = constants_1[0][1]
		u1_fit = constants_1[0][2]
		v1_fit = constants_1[0][3]
	
		constants2 = curve_fit(profile,ydataset,zdataset)
		y2_fit = constants2[0][0]
		z2_fit = constants2[0][1]
		v2_fit = constants2[0][2]
		w2_fit = constants2[0][3]
	
		# plotting parametric curve
		y_parametric = np.linspace(0, 20, 100)
	
		t = (y1_fit-y_parametric)/v1_fit
		x_parametric = x1_fit + u1_fit*t
	
		t = (y2_fit-y_parametric)/v2_fit
		z_parametric = z2_fit + w2_fit*t - 0.5*9.81*(t**2)
	
		ax.plot(x_parametric, y_parametric, z_parametric, label='parametric curve')
	 
	 
def linear(x,m,c):
	return m*x+c
 
def parabolic(x, a, b, c):
	return a*(x**2) + b*x+c

def plane(x, x0, y0, u0, v0):
	t = (x-x0)/u0
	y = y0-v0*t
	return y

def profile(y, y0, z0, v0, w0):
	t = (y0-y)/v0
	z = z0 + w0*t - 0.5*9.81*(t**2)
	return z	

def callback(data):
	global samples
	
	rospy.loginfo(f'[CURVE FITTING COUNTER] {samples.shape[0]}')
	
	extracts = str(data).split(' ', 2)[1]
	#rospy.loginfo(extracts)
	stripped_data = extracts.strip('"')
	#rospy.loginfo(stripped_data)
	extracts = stripped_data.split('|', 4)
	#rospy.loginfo(extracts)
			
	if samples.shape[1]==0:
		samples = np.array([[float(extracts[0]), float(extracts[1]), float(extracts[2]), float(extracts[3])]])
	else:
		samples = np.vstack([samples, [[float(extracts[0]), float(extracts[1]), float(extracts[2]), float(extracts[3])]]])
    				    
	rospy.loginfo(samples)
	update_figure()

      
def receive_message():
	global pub
	# Tells rospy the name of the node.
	# Anonymous = True makes sure the node has a unique name. Random
	# numbers are added to the end of the name. 
	rospy.init_node('curve_estimator', anonymous=True)
	
	# Node is subscribing to the sample_data topic
	rospy.Subscriber('samples', String, callback)
	
	# spin() simply keeps python from exiting until this node is stopped
	#rospy.spin()
	plt.show(block=True)
  
if __name__ == '__main__':
	receive_message()
