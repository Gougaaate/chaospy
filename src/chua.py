import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


# Define the function Phi(x)
def Phi(x, m0=15, m1=0.2):
    return x + m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))

# Define the differential equations
def dxdt(x, y, alpha=15, m0=15, m1=0.2):
    return alpha * (y - Phi(x, m0, m1))

def dydt(x, y, z):
    return x - y + z

def dzdt(y, beta=0.2):
    return -beta * y

def euclidean_distance(x, y, z, x1, y1, z1):
    """Compute the euclidean distance between the points of two lists of points in 3D space"""
    dist = []
    for i in range(len(x)):
        dist.append(np.sqrt((x[i]-x1[i])**2 + (y[i]-y1[i])**2 + (z[i]-z1[i])**2))
    return dist

def euler(x, y, z, t, dt, alpha=15, beta=0.2, m0=15, m1=0.2):
    """perform the Euler integration
    and store the values of x,y,z in three arrays"""
    X, Y, Z = [], [], []
    for i in range(1, len(t)):
        x[i] = x[i-1] + dt * dxdt(x[i-1], y[i-1], alpha)
        y[i] = y[i-1] + dt * dydt(x[i-1], y[i-1], z[i-1])
        z[i] = z[i-1] + dt * dzdt(y[i-1], beta)
        X.append(x[i])  
        Y.append(y[i])
        Z.append(z[i])
    return X, Y, Z



# Create figure and 3D axis
def init_figure(alpha, beta, m0, m1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Trajectory of the point (x, y, z), alpha = {alpha}, beta = {beta}, m0 = {m0}, m1 = {m1}')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    return fig, ax

def plot_distance(X, Y, Z, X1, Y1, Z1,t):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, euclidean_distance(X, Y, Z, X1, Y1, Z1))
    ax.set_xlabel('t')
    ax.set_ylabel('Euclidean distance')
    ax.set_title('Euclidean distance between the two trajectories')
    plt.show()

# Initialize plot objects

# Function to update the plot for each frame of the animation

if __name__ == "__main__":

    # Define the parameters
    alpha = 10
    beta = 12
    m0 = -8/7
    m1 = -5/7

    # Define the time step and duration
    dt = 0.01
    t = np.arange(0, 100, dt)

    # Initialize arrays to store the coordinates
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    x1 = np.zeros_like(t)
    y1 = np.zeros_like(t)
    z1 = np.zeros_like(t)

    
    # Initial conditions
    x[0] = 1.0
    y[0] = 1.0
    z[0] = 1.0
    x1[0] = 1.01
    y1[0] = 1.01
    z1[0] = 1.01


    X, Y, Z = euler(x, y, z, t, dt, alpha, beta, m0, m1)
    X1, Y1, Z1 = euler(x1, y1, z1, t, dt, alpha, beta, m0, m1)

    fig, ax = init_figure(alpha, beta, m0, m1)
    point, = ax.plot([], [], [], 'bo', markersize=5)  # point marker
    trace, = ax.plot([], [], [], 'r-', alpha=0.5)    # trace line

    def update(frame):
        point.set_data(x[frame], y[frame])
        point.set_3d_properties(z[frame])
        trace.set_data(x[:frame+1], y[:frame+1])
        trace.set_3d_properties(z[:frame+1])
        return point, trace

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)

    plt.show()
    plot_distance(X, Y, Z, X1, Y1, Z1, t[:-1])