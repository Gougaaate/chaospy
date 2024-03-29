import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

def logistic_map(t0, tf, points_number):
    """Computes the logistic map for a given range of time and a given number of points.
    """
    P = np.linspace(t0, tf, points_number)
    X, Y = [], []

    for u in P:
        X.append(u)
        m = np.random.random()
        for n in range(400):
            m = (u * m) * (1 - m)

        Y.append(m)
    plt.plot(X, Y, ls='', marker=',')
    plt.xlabel("Value of \u03BC")
    plt.title("Logistic map")
    plt.show()

def logistic(t0, tf, m):
    """Computes values of the logistic function given the m parameter.
    Writes them in a file.
    """
    t = np.linspace(t0, tf, 10000)
    x = 0.75
    X = [x]
    for i in range(len(t)):
        x = m * x * (1 - x)
        X.append(x)
    with open ("../data/logistic.txt", "w") as f:
        f.write(str(m) + "\n")
        for i in range(len(X)):
            f.write(str(X[i]) + "\n")
    return X

def plot_logistic(X, mu):
    """Plots the logistic function given the values of X.
    """
    plt.plot(X[:150])
    plt.title(f"Logistic function, \u03BC = {mu}")
    plt.ylabel("Value of x_n")
    plt.xlabel("Value of n")
    plt.show()

if __name__ == "__main__":
  
    mu = 3.82
    # X = logistic(0, 500, mu)
    # plot_logistic(X, mu)
    logistic_map(2, 4, 100000)
    print("end of the program") 
    