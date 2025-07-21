import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import pyplot as mp

def gaussian(x, mu, sig):
    return (
      -sig*(1 -np.exp(-np.power(x,2)))
    )


x_values = np.linspace(-3, 3, 120)
for mu, sig in [(0, 0.7)]:
    mp.plot(x_values, -gaussian(x_values, mu, sig))

mp.show()