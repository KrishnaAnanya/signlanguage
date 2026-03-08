import numpy as np
from collections import Counter

x=np.load("X.npy")
y=np.load("Y.npy")
print(x.shape)
print(y.shape)
z=Counter(y)
print(z)