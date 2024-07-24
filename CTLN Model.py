import numpy as np
import itertools as itt
import pandas as pd
import scipy as scp
import matplotlib.pyplot as plt
import random as rd

# parameters for eps, del, the
e = 0.25
d = 0.5
t = 1

# creating powersets function
def powerset(j):
    ps = [] # power set of parent set
    xs = np.arange(1,j+1)
    for i in range(0, j+1):
        for subset in itt.combinations(xs, i):
            ps.append(subset)
    return(ps)

# generating weight matrix function
def weight_matrix(matrix_size):
    mat = np.zeros((matrix_size, matrix_size))

    possible_vals = (-2,-1,1,2)
    weights = (0.2,0.3, 0.3,0.2)

    for i in range(matrix_size):
        for j in range(matrix_size):
            if i == j:
                continue
            else:
                mat[i,j] = rd.choices(possible_vals, weights)[0]

    for i in range(matrix_size):
        for j in range(matrix_size):
            if mat[i,j] == -2:
                mat[i,j] = -1-d
                mat[j,i] = -1-d
            if mat[i,j] == 2:
                mat[i,j] = -1+e
                mat[j,i] = -1+e
            if mat[i,j] == -1:
                mat[i,j] = -1-d
            if mat[i,j] == 1:
                mat[i,j] = -1+e

    print(mat)
    return(mat)

# checking fixed points

def check_fp(weight_matrix, fixed_point, theta, theta_sig):
    checker = []
    for i in range(matrix_size):
        Wx = weight_matrix[i,:]@fixed_point + theta[i]
        # print(Wx)
        if theta_sig[i] > 0:
            if Wx > 0:
                checker.append(1)
            else:
                checker.append(0)
        else:
            if Wx > 0:
                checker.append(0)
            else:
                checker.append(1)
    return(checker)

#input weight matrix
matrix_size = int(input('matrix size?\n'))
W = weight_matrix(matrix_size)
#input powerset
ps = powerset(matrix_size)
# print(f'The relevant values of sigma are {ps}')
# print(W)

# dictionary for fp supports

supports = {}

# finding fixed points

A = np.copy(W)
for sigma in ps: # index of entry in power set
    W = np.copy(A) # weight matrix
    theta = np.ones((matrix_size, 1)) # theta
    t_s = np.copy(theta) # theta_sigma
    # print(ps.index(sigma))
    deac_neur = ps[ps.index(sigma)] #deactivated neurons
    if sigma:
        for iter in deac_neur:
            W[iter-1, :] = 0
            inverse = np.linalg.inv(np.identity(matrix_size) - W)   
            t_s[iter-1, :] = 0
            P = np.matmul(inverse, t_s)
        # print(P)
        checker = check_fp(A, P, theta, t_s)

    else:
        inverse = np.linalg.inv(np.identity(matrix_size) - W)
        P = np.matmul(inverse, t_s)
        # print(P)
        checker = check_fp(A, P, theta, t_s)

    if np.all(checker) == True:
        supports[tuple(np.setdiff1d(np.arange(1, matrix_size+1), deac_neur))] = tuple(P)
print(supports)
# print(W_s)

# simulate ODE
# generate random matrix
# A - np.diag(np.diag(A))

# simulating ODE

def sys(t, x):
    # x1, x2 = x    
    dxdt = (-x + np.maximum(0, A@x + 1))
    return dxdt

x0 = np.random.rand(matrix_size, 1)
time = [0, 100]

x = scp.integrate.solve_ivp(sys, time, x0.flatten(), dense_output=True)
t = np.linspace(0,25, 101)

plot1 = plt.plot(t, (x.sol(t)).T)
plt.show()
plt.xlim([0,6])
plt.ylim([0,6])
plot2 = plt.plot(x.sol(t)[0], x.sol(t)[1])
plt.show()

print(pd.DataFrame.from_dict(supports, orient='index'))