import numpy as np

z = np.random.normal(size=[25, 100])

np.save("FixNoise.npy",z)
