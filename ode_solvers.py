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


def adams_moulton_2(dydt, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    dt = t[1] - t[0]
    y[1] = y[0] + dt * dydt(y[0])  # Use Euler's method for first step
    for i in range(2, len(t)):
        # Estimate with 2-step Adams-Bashforth method
        y_ab2 = y[i-1] + dt / 2 * (3 * dydt(y[i-1]) - dydt(y[i-2]))
        # Correct with 2-step Adams-Moulton method
        y[i] = y[i-1] + dt / 2 * (dydt(y[i-1]) + dydt(y_ab2))
    return y


def gears_1(f, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    dt = t[1] - t[0]

    # Define the residual function for the Newton-Raphson method
    def residual(y_new, y_old):
        return y_new - y_old - dt * f(y_new)

    # Define the Jacobian of the residual function for the Newton-Raphson method
    def jacobian(y_new):
        eps = 1e-8  # Small perturbation for numerical differentiation
        return 1 - dt * (f(y_new + eps) - f(y_new)) / eps

    # Perform the integration
    for i in range(1, len(t)):
        # Initial guess for the Newton-Raphson method
        y_guess = y[i-1]

        # Newton-Raphson method
        for j in range(100):
            dy = residual(y_guess, y[i-1]) / jacobian(y_guess)
            y_guess -= dy
            if abs(dy) < 1e-10:
                break

        y[i] = y_guess

    return y


def midpoint_method(f, y0, t0, t1, num_steps):
    y = np.zeros(num_steps)
    y[0] = y0
    dt = (t1 - t0) / (num_steps - 1)
    t = t0
    for i in range(num_steps - 1):
        k1 = f(t, y[i])
        k2 = f(t + dt / 2, y[i] + dt / 2 * k1)
        y[i + 1] = y[i] + dt * k2
        t += dt
    return y


def bulirsch_stoer_step(f, y, t, dt):
    n_steps = 1
    y_next_two = midpoint_method(f, y, t, dt, n_steps * 2)
    error = 1e10  # Arbitrary large value
    while error > 1e-10:
        n_steps *= 2
        y_next = y_next_two
        y_next_two = midpoint_method(f, y, t, dt, n_steps * 2)
        error = abs((y_next_two - y_next) / y_next_two)
    return y_next_two


def bulirsch_stoer(f, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        y[i + 1] = bulirsch_stoer_step(f, y[i], t[i], dt)
    return y


def heun(dydt, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    dt = t[1] - t[0]
    for i in range(0, len(t) - 1):
        # Predictor step
        y_star = y[i] + dt * dydt(y[i])
        # Corrector step
        y[i + 1] = y[i] + dt / 2 * (dydt(y[i]) + dydt(y_star))
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


# Define the differential equation

def dydt(y):
    return y


# Set initial condition and time step
y0 = 1
t = np.arange(0, 2, 0.01)

# 2-step Adams-Moulton method
y_am2 = adams_moulton_2(dydt, y0, t)

# Plot results
plt.plot(t, y_am2, label='Adams-Moulton 2')
plt.plot(t, np.exp(t), label='Analytical')
plt.legend()
plt.show()


# Define the differential equation

def dydt(y):
    return -20 * y  # A stiff differential equation


# Set initial condition and time step
y0 = 1
t = np.arange(0, 2, 0.01)

# 1st order Gear's method
y_g1 = gears_1(dydt, y0, t)

# Plot results
plt.plot(t, y_g1, label='Gear 1')
plt.legend()
plt.show()


# Define the differential equation

def dydt(t, y):
    return y


# Set initial condition and time step
y0 = 1
t = np.arange(0, 2, 0.01)

# Bulirsch-Stoer method
y_bs = bulirsch_stoer(dydt, y0, t)

# Plot results
plt.plot(t, y_bs, label='Bulirsch-Stoer')
plt.plot(t, np.exp(t), label='Analytical')
plt.legend()
plt.show()


# Define the differential equation

def dydt(y):
    return y


# Set initial condition and time step
y0 = 1
t = np.arange(0, 2, 0.01)

# Heun's method
y_heun = heun(dydt, y0, t)

# Plot results
plt.plot(t, y_heun, label='Heun')
plt.plot(t, np.exp(t), label='Analytical')
plt.legend()
plt.show()
