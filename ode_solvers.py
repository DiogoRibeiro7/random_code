import numpy as np
import matplotlib.pyplot as plt


def euler_method(dydt, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        y[i] = y[i-1] + dt * dydt(y[i-1])
    return y


def runge_kutta_4(dydt, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        k1 = dt * dydt(y[i-1])
        k2 = dt * dydt(y[i-1] + 0.5 * k1)
        k3 = dt * dydt(y[i-1] + 0.5 * k2)
        k4 = dt * dydt(y[i-1] + k3)
        y[i] = y[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


def leapfrog_method(f, x0, v0, t):
    # Initialize position and velocity arrays
    x = np.zeros(len(t))
    v = np.zeros(len(t))

    x[0] = x0
    v[0] = v0

    dt = t[1] - t[0]

    # Half step velocity
    v_half = v[0] + 0.5 * dt * f(x[0])

    for i in range(1, len(t)):
        # Full step position
        x[i] = x[i-1] + dt * v_half

        # Full step velocity
        v_full = v_half + 0.5 * dt * f(x[i])

        # Prepare for next iteration
        if i < len(t) - 1:
            v_half = v_full + 0.5 * dt * f(x[i])

        v[i] = v_full

    return x, v


def adams_bashforth_2(dydt, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    dt = t[1] - t[0]
    y[1] = y[0] + dt * dydt(y[0])  # Use Euler's method for first step
    for i in range(2, len(t)):
        y[i] = y[i-1] + dt / 2 * (3 * dydt(y[i-1]) - dydt(y[i-2]))
    return y



# Define the differential equation
def dydt(y):
    return y


# Set initial condition and time step
y0 = 1
t = np.arange(0, 2, 0.01)

# Euler's method
y_euler = euler_method(dydt, y0, t)

# 4th order Runge-Kutta method
y_rk4 = runge_kutta_4(dydt, y0, t)

# Plot results
plt.plot(t, y_euler, label='Euler')
plt.plot(t, y_rk4, label='Runge-Kutta 4')
plt.plot(t, np.exp(t), label='Analytical')
plt.legend()
plt.show()


# Define the force function

def f(x):
    return -x  # Hooke's law


# Set initial condition and time step
x0 = 1
v0 = 0
t = np.arange(0, 10, 0.01)

# Leapfrog method
x, v = leapfrog_method(f, x0, v0, t)

# Plot results
plt.plot(t, x, label='Position')
plt.plot(t, v, label='Velocity')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation
def dydt(y):
    return y

# Set initial condition and time step
y0 = 1
t = np.arange(0, 2, 0.01)

# 2-step Adams-Bashforth method
y_ab2 = adams_bashforth_2(dydt, y0, t)

# Plot results
plt.plot(t, y_ab2, label='Adams-Bashforth 2')
plt.plot(t, np.exp(t), label='Analytical')
plt.legend()
plt.show()
