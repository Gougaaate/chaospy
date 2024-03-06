import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Constants
G = 9.81  # Acceleration due to gravity (m/s^2)
L1 = 1  # Length of pendulum 1 (m)
L2 = .5  # Length of pendulum 2 (m)
M1 = 1.0  # Mass of pendulum 1 (kg)
M2 = 1.0  # Mass of pendulum 2 (kg)

# Function to compute the double pendulum equations of motion
def double_pendulum_equations(state, t):
    theta1, omega1, theta2, omega2 = state
    
    # Derivatives
    dtheta1_dt = omega1
    domega1_dt = (-G * (2 * M1 + M2) * np.sin(theta1) - M2 * G * np.sin(theta1 - 2 * theta2) - 2 * np.sin(theta1 - theta2) * M2 * (omega2**2 * L2 + omega1**2 * L1 * np.cos(theta1 - theta2))) / (L1 * (2 * M1 + M2 - M2 * np.cos(2 * theta1 - 2 * theta2)))
    dtheta2_dt = omega2
    domega2_dt = (2 * np.sin(theta1 - theta2) * (omega1**2 * L1 * (M1 + M2) + G * (M1 + M2) * np.cos(theta1) + omega2**2 * L2 * M2 * np.cos(theta1 - theta2))) / (L2 * (2 * M1 + M2 - M2 * np.cos(2 * theta1 - 2 * theta2)))
    
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

# Function to animate the double pendulum
def animate_double_pendulum(i, frames1, frames2):
    theta1_vals1 = frames1[:, 0]
    theta2_vals1 = frames1[:, 2]
    theta1_vals2 = frames2[:, 0]
    theta2_vals2 = frames2[:, 2]
    
    x1_1 = L1 * np.sin(theta1_vals1)
    y1_1 = -L1 * np.cos(theta1_vals1)
    x2_1 = x1_1 + L2 * np.sin(theta2_vals1)
    y2_1 = y1_1 - L2 * np.cos(theta2_vals1)
    
    x1_2 = L1 * np.sin(theta1_vals2) + 3.0
    y1_2 = -L1 * np.cos(theta1_vals2)
    x2_2 = x1_2 + L2 * np.sin(theta2_vals2)
    y2_2 = y1_2 - L2 * np.cos(theta2_vals2)
    
    line1.set_data([0, x1_1[i], x2_1[i]], [0, y1_1[i], y2_1[i]])
    line2.set_data([3.0, x1_2[i], x2_2[i]], [0, y1_2[i], y2_2[i]])
    
    trace1.set_data(x2_1[:i], y2_1[:i])
    trace2.set_data(x2_2[:i], y2_2[:i])
    
    return line1, line2, trace1, trace2

def plot_theta2_at_end(frames1, t):
    plt.figure()
    plt.plot(t[:1000], (frames1[:,2][:1000])%(2*np.pi) * 180/np.pi)
    plt.xlabel('time (s)')
    plt.ylabel('\u03B8 2 (degrees)')
    plt.title('Pendulum angle over time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Initial conditions
    initial_state1 = [ np.pi/2 + 0.001, 0, np.pi, 0]  # [theta1, omega1, theta2, omega2]
    initial_state2 = [ np.pi/2, 0, np.pi, 0]  # [theta1, omega1, theta2, omega2]

    # Time points
    t = np.linspace(0, 100, 5000)

    # Solve the differential equations for each pendulum
    frames1 = odeint(double_pendulum_equations, initial_state1, t)
    frames2 = odeint(double_pendulum_equations, initial_state2, t)
    print(frames1)

    # Set up the figure and axes
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 5)
    ax.set_ylim(-2, 2)

    # Plot the double pendulum and traces
    line1, = ax.plot([], [], 'o-', lw=2, color='blue')
    line2, = ax.plot([], [], 'o-', lw=2, color='red')
    trace1, = ax.plot([], [], '-', lw=1, alpha=0.5, color='blue')
    trace2, = ax.plot([], [], '-', lw=1, alpha=0.5, color='red')

    # Create the animation
    ani = FuncAnimation(fig, animate_double_pendulum, frames=len(frames1), interval=20, blit=True, fargs=(frames1, frames2))

    # Show the animation
    plt.show()
    plot_theta2_at_end(frames1, t)
