import numpy as np
import random


def monte_carlo_integration(f, a, b, n):
    return (b - a) / float(n) * sum(f(random.uniform(a, b)) for _ in range(n))


def gaussian_quadrature(f, a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return ((b - a) / 2) * sum(wi * f((b - a) / 2 * xi + (b + a) / 2) for xi, wi in zip(x, w))

def simpsons_rule(f, a, b, n):
    dx = (b - a) / n
    return (f(a) + 4 * sum(f(a + dx * i) for i in range(1, n, 2)) + 2 * sum(f(a + dx * i) for i in range(2, n, 2)) + f(b)) * dx / 3

def trapezoidal_rule(f, a, b, n):
    dx = (b - a) / n
    return (f(a) + 2 * sum(f(a + i * dx) for i in range(1, n)) + f(b)) * dx / 2


def trapezoidal_rule(f, a, b, N):
    h = (b - a) / N
    s = 0.5 * f(a) + 0.5 * f(b) + sum(f(a + i * h) for i in range(1, N))
    return s * h

def romberg_integration(f, a, b, tol=1e-6, max_iter=1000):
    R = [[0]]
    R[0][0] = trapezoidal_rule(f, a, b, 1)
    n = 1

    while n < max_iter:
        h = (b - a) / 2.0 ** n
        R.append([0] * (n + 1))
        # Compute trapezoidal rule for next level of subdivision
        R[n][0] = 0.5 * R[n - 1][0] + h * sum(f(a + (2 * k - 1) * h) for k in range(1, 2 ** (n - 1) + 1))

        # Compute extrapolation estimates
        for m in range(1, n + 1):
            R[n][m] = R[n][m - 1] + 1.0 / (4 ** m - 1) * (R[n][m - 1] - R[n - 1][m - 1])

        # Check if the last two estimates are close enough
        if abs(R[n][n - 1] - R[n][n]) < tol:
            return R[n][n]

        n += 1

    raise Exception(f"Romberg integration did not converge within {max_iter} iterations")

# example usage
def f(x):
    return x**2

result = romberg_integration(f, 0, 1)
print(result)

def riemann_sum_midpoint(f, a, b, n):
    dx = (b - a) / n
    return sum(f(a + dx * (i + 0.5)) for i in range(n)) * dx

def adaptive_quadrature(f, a, b, tol=1e-6):
    # Simpson's rule
    h = b - a
    s = h / 6 * (f(a) + 4 * f(a + h / 2) + f(b))

    def adaptive_aux(f, a, b, s, tol, h):
        c = a + h / 2
        left = h / 12 * (f(a) + 4 * f(a + h / 4) + f(c))
        right = h / 12 * (f(c) + 4 * f(c + h / 4) + f(b))
        if abs(left + right - s) < 15 * tol:
            return left + right + (left + right - s) / 15
        else:
            return adaptive_aux(f, a, c, left, tol / 2, h / 2) + adaptive_aux(f, c, b, right, tol / 2, h / 2)

    return adaptive_aux(f, a, b, s, tol, h)

# example usage


def f(x):
    return x**2


result = adaptive_quadrature(f, 0, 1)
print(result)


def booles_rule(f, a, b, n):
    h = (b - a) / n
    return 2 * h / 45 * (7*f(a) + 32 * sum(f(a + i * h) for i in range(1, n, 4)) + 12 * sum(f(a + i * h) for i in range(2, n, 4)) + 32 * sum(f(a + i * h) for i in range(3, n, 4)) + 7 * sum(f(a + i * h) for i in range(4, n, 4)) + 7 * f(b))
