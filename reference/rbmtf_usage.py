# %%
from rbmtf import RBM
import numpy as np 


# %%
test = np.array([[0,1,1,0], [0,1,0,0], [0,0,1,1]])
rbm = RBM(4, 3, 0.1, 100)
rbm.train(test)


# %%
