import numpy as np
import itertools as itt

# parameters for eps, del, the
e = 0.25
d = 0.5
t = 1

# creating powersets function
def powerset(j):
    xs = [] # parent set
    ps = [] # power set of parent set
    for i in range(j):
        xs.append(i+1)
    for i in range(0, j+1):
        for subset in itt.combinations(xs, i):
            ps.append(subset)
    return(ps)

# generating weight matrix function
def weight_matrix(r, c):
    w = []
    print('enter entries row-wise')
    for i in range(r):
        indiv_row = []    
        for j in range(c):
            indiv_row.append(int(input()))
        w.append(indiv_row)
    mat_w = np.array(w)

    w_mod = []
    for row in mat_w:
        w_mod_r = []
        for ent in row:
            if ent == 1:
                w_mod_r.append(-1+e)
                continue
            if ent == -1:
                w_mod_r.append(-1-d)
                continue
            else:
                w_mod_r.append(0)
        # print(w_mod_r)
        w_mod.append(w_mod_r)
    # mat_w_mod = np.array(w_mod)
    mat_w_mod = np.array(w_mod)
    return(mat_w_mod)


#input weight matrix
matrix_size = int(input('matrix size?\n'))
W = weight_matrix(matrix_size, matrix_size)
#input powerset
ps = powerset(matrix_size)
# print(f'The relevant values of sigma are {ps}')
print(W)

# finding fixed points

A = np.copy(W)
for sigma in ps: # index of entry in power set
    W = np.copy(A) # weight matrix
    t_s = np.ones((matrix_size, 1)) # theta_sigma
    print(ps.index(sigma))
    deac_neur = ps[ps.index(sigma)] #deactivated neurons
    if sigma:
        for iter in deac_neur:
            W[iter-1, :] = 0
            inverse = np.linalg.inv(np.identity(matrix_size) - W)   
            t_s[iter-1, :] = 0
            P = np.matmul(inverse, t_s)
        print(P)
        for i in range(matrix_size):
            s=[]
            for j in range(matrix_size):
                # print(W[i,j])
                # print(P[j])
                Wx = A[i,j]*P[j]
                s.append(Wx)
                Wx = 0
            check = sum(s) + t_s[i]
            print(check)


    else:
        inverse = np.linalg.inv(np.identity(matrix_size) - W)
        P = np.matmul(inverse, t_s)
        print(P)
        # what is even going on here
        for i in range(matrix_size):
            s=[]
            for j in range(matrix_size):
                # print(W[i,j])
                # print(P[j])
                Wx = A[i,j]*P[j]
                s.append(Wx)
                Wx = 0
            check = sum(s) + t_s[i]
            print(check)
# print(W_s)


# mat_inverse_fp = np.linalg.inv(np.identity(matrix_size) - mat_w_mod)
# print(mat_inverse_fp)