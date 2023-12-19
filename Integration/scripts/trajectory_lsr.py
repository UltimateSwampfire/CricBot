#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PointStamped
import numpy as np
# import time
# import random

#x = acceleration*0.5*t*t + a*t + b represents 1D motion under constant acceleration
#solving for a and b in the above equation eqn using least square regression
class lsr1DTrajectory:
	def __init__(self,acc=0.0):
		self.x = []
		self.y = []
		self.a = 0.0
		self.b = 0.0
		self.acc = acc
		self.sumX = 0.0
		self.sumY = 0.0
		self.sumXY = 0.0
		self.sumX2 = 0.0
		self.sumX3 = 0.0
		self.n = 0.0

	def find_coordinate(self,t):		
		return ((self.acc*0.5*t*t) + (self.a*t) + self.b)

	def find_time(self,x):
		if self.acc == 0:
			return (x-self.b)/self.a
		else:
			a1 = self.acc*0.5
			b1 = self.a
			c1 = self.b-x
			return ( -b1 + ( ( (b1*b1) - (4*a1*c1) )**0.5 ) )/(2*a1)

	def calculateFromPoint(self,x,y,update=True):
		self.x.append(x)
		self.y.append(y)
		self.n+=1.0

		self.sumX+=x
		self.sumY+=y
		self.sumXY+=x*y
		self.sumX2+=x*x
		self.sumX3+=x*x*x		

		if self.n>=2:
			A = np.zeros((2,2))
			B = np.zeros((2,1))		

			A[0][0] = self.sumX2
			A[0][1] = self.sumX
			A[1][0] = self.sumX
			A[1][1] = self.n
			B[0][0] = self.sumXY-(self.acc*0.5*self.sumX3)
			B[1][0] = self.sumY-(self.acc*0.5*self.sumX2)

			ans = np.linalg.inv(A).dot(B)
			ans = [ans[0][0],ans[1][0]]
			if update:
				self.a = ans[0]
				self.b = ans[1]
			return ans

	#calculate a and b using data passed as parameters
	def calculateFromList(self,x,y,update=True):
		n = len(x)
		z = np.zeros((n,2))
		B = np.zeros((2,1))

		for i in range(n):
			z[i] = [x[i],1]
			B[0][0] += (x[i]*y[i]) - 0.5*self.acc*(x[i]*x[i]*x[i])
			B[1][0] += y[i] - 0.5*self.acc*(x[i]*x[i])

		ans = np.linalg.inv(z.T.dot(z)).dot(B)
		ans = [ans[0][0],ans[1][0]]
		if update:
			self.a = ans[0]
			self.b = ans[1]
		return ans


ball_coordinates = []

trajX = lsr1DTrajectory(0.0)
trajY = lsr1DTrajectory(0.0)
trajZ = lsr1DTrajectory(-9.8)

def call_back(msg):
	global ball_coordinates,trajX,trajY,trajZ
	ball_coordinates.append(msg)
	tnow = (msg.header.stamp.secs*1000000000 + msg.header.stamp.nsecs)/1000000000.0

	trajX.calculateFromPoint(tnow,msg.point.x)
	trajY.calculateFromPoint(tnow,msg.point.y)
	trajZ.calculateFromPoint(tnow,msg.point.z)

	if len(ball_coordinates)>=2:
		tpred = trajX.find_time(1.5)
		print(tpred,[trajX.find_coordinate(tpred),trajY.find_coordinate(tpred),trajZ.find_coordinate(tpred)])


def listener():
	print("trajectory_lsr node initiated")
	rospy.init_node('trajectory_lsr')
	coord_sub = rospy.Subscriber('ball_coordinate',PointStamped,call_back)
	rospy.spin()


if __name__ == "__main__":
	try:
		listener()
	except rospy.ROSInterruptException:
		pass


