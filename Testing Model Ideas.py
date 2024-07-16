import numpy as np
import itertools as itt

# parameters for eps, del, the
e = 0.25
d = 0.5
t = 1

# creating powersets function
def powerset(int):
    xs = [] # parent set
    ps = [] # power set of parent set
    for i in range(int):
        xs.append(i+1)
    for i in range(0, int+1):
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
    # print(w)
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
    mat_w_mod = np.array(w_mod)
    return(mat_w_mod)


#input weight matrix
matrix_size = int(input('matrix size?\n'))
mat_w_mod = weight_matrix(matrix_size, matrix_size)
#input powerset
ps = powerset(matrix_size)
print(f'The relevant values of sigma are {ps}')

# finding fixed points

eval_mat = []
for subset in ps:
    eval_mat_row = []
    for entry in subset:
        for row in range(len(mat_w_mod)):
            if entry != row+1:
                for i in range(matrix_size):
                    eval_mat_row.append(0)
            else:
                continue
        eval_mat.append(eval_mat_row)

print(eval_mat)

# mat_inverse_fp = np.linalg.inv(np.identity(matrix_size) - mat_w_mod)
# print(mat_inverse_fp)