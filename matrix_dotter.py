import numpy as np

matrix_A = np.array([[0, 1, 0, 0, 1],
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0]])

matrix_B = np.array([0,0,0,1])

result = np.dot(matrix_A, matrix_B)
print(result)

