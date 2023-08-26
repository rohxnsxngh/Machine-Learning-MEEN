print("hello world")
import numpy as np
u = np.array([1, 2, 3], dtype='float')
v = np.array([4, 5, 6], dtype='float')
w = np.outer(u, v)
print(w)