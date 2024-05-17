import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial


lam = 10.0
x = np.arange(0, 25)
y = np.exp(-lam) * np.power(lam, x) / factorial(x)
# y = np.array([np.exp(-float(lam)) * float(lam)**float(xi) / factorial(xi) for xi in x])
print(np.exp(-10)*10**11/factorial(11))
print(x)
print(y)
print(y[10])
plt.plot(x,y, 'o-')
plt.show()
