import numpy as np

eval_mat = []
ps = [[1],[2], [1,2]]
mat_w_mod = [[0,1],[-1,0]]
matrix_size = 2

for subset in ps:
    if len(subset) < matrix_size:
        for entry in subset:
            for row in range(len(mat_w_mod)):
                eval_mat_row = []
                print(entry, row+1)
                if entry == row+1:
                    # print('true')
                    eval_mat_row = mat_w_mod[row]
                    print(eval_mat_row)
                else:
                    # print('false')
                    for i in range(matrix_size):
                        eval_mat_row.append(0)
                eval_mat.append(eval_mat_row)
    if len(subset) == matrix_size:
        for entry in mat_w_mod:
            

        

print(eval_mat)