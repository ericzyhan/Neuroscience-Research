# Random graph generation

import numpy as np
val = int(input('size of matrix?'))

graph_matrix = np.random.rand(val,val)
print(graph_matrix)
graph_matrix_thresholded=[]

for row in graph_matrix:
    threshold_row = []
    for threshold_value in row:
        if threshold_value >= 0.4:
            threshold_row.append(threshold_value)
        else:
            threshold_row.append('0')
    graph_matrix_thresholded.append(threshold_row)

print(graph_matrix_thresholded)

