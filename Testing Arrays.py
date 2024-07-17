import numpy as np
A = np.random.rand(5,5)
# print(A)

W = np.copy(A)
ps = [[1],[2],[1,2]]

for entry in ps:
    A = np.copy(W)
    print(ps.index(entry))
    i = ps[ps.index(entry)]
    for iter in i:
        A[iter-1, :] = 0
        
    print(A)



# print(A)