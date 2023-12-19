from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


x = [1,2,3,4,5]
y = [1.9,2.6,3.1,3.4,3.5]

def quad(x,a,b,c,d):

    return a*x*x + b*x + c + d

def profile(y, y0, z0, v0, w0):
	t = (y0-y)/v0
	z = z0 + w0*t - 0.5*9.81*(t**2)
	return z	

p = curve_fit(quad,x,y)

x_line = np.arange(min(x),max(x),100)
y_line = profile(x_line,p[0][0],p[0][1],p[0][2],p[0][3])
print(p[0])

plt.plot(x_line,y_line)
plt.scatter(x,y)
plt.show()

