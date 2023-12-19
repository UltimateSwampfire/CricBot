import numpy as np



x = np.linspace(1,40,40)

y = x*(40-x)
y = y+(2*np.random.rand(len(y))-1)*20

