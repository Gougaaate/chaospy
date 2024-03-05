import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Define the parameters
alpha = 1.0
beta = 1.0
m0 = 1.0
m1 = 1.0

# Define the time step and duration
dt = 0.01
t = np.arange(0, 10, dt)

# Define the function Phi(x)
def Phi(x):
    return x + m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))

# Define the differential equations
def dxdt(x, y):
    return alpha * (y - Phi(x))

def dydt(x, y, z):
    return x - y + z

def dzdt(y):
    return -beta * y


# Initialize arrays to store the coordinates
x = np.zeros_like(t)
y = np.zeros_like(t)
z = np.zeros_like(t)

# Initial conditions
x[0] = 1.0
y[0] = 1.0
z[0] = 1.0

# Perform Euler integration
for i in range(1, len(t)):
    x[i] = x[i-1] + dt * dxdt(x[i-1], y[i-1])
    y[i] = y[i-1] + dt * dydt(x[i-1], y[i-1], z[i-1])
    z[i] = z[i-1] + dt * dzdt(y[i-1])

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trajectory of the point (x, y, z) with trace')

# Initialize plot objects
point, = ax.plot([], [], [], 'bo', markersize=5)  # point marker
trace, = ax.plot([], [], [], 'r-', alpha=0.5)    # trace line

# Function to update the plot for each frame of the animation
def update(frame):
    point.set_data(x[frame], y[frame])
    point.set_3d_properties(z[frame])
    trace.set_data(x[:frame+1], y[:frame+1])
    trace.set_3d_properties(z[:frame+1])
    return point, trace

# Create animation
ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)

plt.show()
