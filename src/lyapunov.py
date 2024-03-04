# File that computes the Lyapunov exponent of a dynamical system
from math import log
import matplotlib.pyplot as plt

def Lyapunov_logistic(file):
    """Computes the Lyapunov exponent of the logistic suite
    """
    with open (file, 'r') as f:
        lines = f.readlines()
        x = []
        for line in lines:
            x.append(float(line))
    n = len(x)
    mu = x[0]
    LE = []
    lambd = 0.0
    for i in range(1, n):
        lambd += (1/i) * log(abs(mu - 2 * mu * x[i]))
        LE.append(lambd)
    plt.plot(LE)
    plt.show()

if __name__ == "__main__":
    Lyapunov_logistic("../data/logistic.txt")