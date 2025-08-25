import numpy as np
import scipy.integrate as scp
import statistics as st
import matplotlib.pyplot as plt
import os

# parameters for eps, del, the
e = 0.25
d = 0.5
t = 1

def square(x):
    return x**2

# generating balanced weight matrix
def weight_matrix_bal(matrix_size,p,q):
    mat = np.zeros((matrix_size, matrix_size))

    # sc_p = (p*10)/matrix_size
    # sc_q = (q*10)/matrix_size

    sc_p = p
    sc_q = q

    # print(sc_p)
    # print(sc_q)

    possible_vals = (-2,-1,1,2)
    weights = (1-sc_p,sc_p*((1-sc_q)/2),sc_p*((1-sc_q)/2),sc_p*sc_q)
    # print(weights)

    for i in range(matrix_size):
        for j in range(matrix_size):
            if i >= j:
                continue
            else:
                mat[i,j] = np.random.choice(possible_vals, 1, p=weights)[0]
    # print(mat)

    for i in range(matrix_size):
        for j in range(matrix_size):
            # if mat[i,j] == -2:
            #     mat[i,j] = -1-d
            #     mat[j,i] = -1-d
            if mat[i,j] == 2:
                mat[i,j] = -1+e
                mat[j,i] = -1+e
            if mat[i,j] == -2:
                mat[i,j] = -1-d
                mat[j,i] = -1-d
            if mat[i,j] == -1:
                mat[i,j] = -1-d
                mat[j,i] = -1+e
            if mat[i,j] == 1:
                mat[i,j] = -1+e
                mat[j,i] = -1-d

    # print(mat)
    return(mat)

def mat_graph(mat, matrix_size):
    mat_graph = mat.copy()

    for i in range(matrix_size):
        for j in range(matrix_size):
            if mat_graph[i,j] == -1-d:
                mat_graph[i,j] = 0
            if mat_graph[i,j] == -1+e:
                mat_graph[i,j] = 1
            # print(i,j)
    return(mat_graph)

# Separate the trial code into a function so we can parallelize it
# trial_number is a dummy argument for pool_trial.map(), it does nothing
def worker_onetrial(mat_size, p, q, trial_number):
    print("Doing trial", trial_number)
    W = weight_matrix_bal(mat_size, p, q)
    A = np.copy(W)  # A is defined locally for stability

    # define the dynamical system
    def sys(t, x):
        x = x.reshape(-1, 1)  # x col vector
        dxdt = -x + np.maximum(0, A @ x + 1)
        return dxdt.flatten()  # flatten for solve_ivp

    # solver
    x0 = np.random.rand(mat_size, 1)
    time = [0, 600]
    sol = scp.solve_ivp(sys, time, x0.flatten(), method='BDF', dense_output=False)

    # fixed-point detection
    final_state = sol.y[:, -1]
    dxdt_final = sys(sol.t[-1], final_state)

    if np.all(np.abs(dxdt_final) < 1e-3):  # check if derivatives are close to zero
        return 1, sol
    else:
        return 0, sol

# Workaround for retrieving sol as a global variable from the module
def get_sol():
    return sol
