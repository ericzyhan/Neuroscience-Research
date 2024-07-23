import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

def sys(t, x):
    # x1, x2 = x    
    dxdt = (-x + np.maximum(0, A@x + 1))
    return dxdt

A = np.array([[0, -1.5], 
              [-0.75, 0]])
x0 = np.array([[4],[1]])
time = [0, 100]

x = scp.integrate.solve_ivp(sys, time, x0.flatten(), dense_output=True)
t = np.linspace(0,25, 101)

print(x)
plot1 = plt.plot(t, (x.sol(t)).T)
plt.show()
plt.xlim([0,6])
plt.ylim([0,6])
plot2 = plt.plot(x.sol(t)[0], x.sol(t)[1])
plt.show()