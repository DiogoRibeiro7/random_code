from typing import Callable, Tuple
import numpy as np
from typing import Callable


def simulated_annealing(objective: Callable[[float], float],
                        initial_state: float,
                        initial_temperature: float,
                        cooling_rate: float,
                        iterations: int) -> Tuple[float, float]:
    """
    Performs the Simulated Annealing algorithm for a given objective function.

    :param objective: The objective function to optimize.
    :type objective: Callable[[float], float]
    :param initial_state: The initial state.
    :type initial_state: float
    :param initial_temperature: The initial temperature.
    :type initial_temperature: float
    :param cooling_rate: The cooling rate.
    :type cooling_rate: float
    :param iterations: The number of iterations to perform.
    :type iterations: int
    :return: The final state and its objective value.
    :rtype: Tuple[float, float]

    Example Usage:
    ----------------
    objective = lambda x: -x**2 + x + 10  # Simple parabolic function with maximum at x=0.5
    initial_state = 0
    initial_temperature = 10
    cooling_rate = 0.99
    iterations = 10000
    state, value = simulated_annealing(objective, initial_state, initial_temperature, cooling_rate, iterations)
    print(f"Final state: {state}")
    print(f"Objective value: {value}")
    """
    current_state = initial_state
    current_temperature = initial_temperature

    for _ in range(iterations):
        proposed_state = current_state + np.random.normal()

        if objective(proposed_state) > objective(current_state):
            current_state = proposed_state
        else:
            acceptance_probability = np.exp(
                (objective(proposed_state) - objective(current_state)) / current_temperature)

            if np.random.rand() < acceptance_probability:
                current_state = proposed_state

        current_temperature *= cooling_rate

    return current_state, objective(current_state)


def fast_simulated_annealing(objective: Callable[[float], float],
                             initial_state: float,
                             initial_temperature: float,
                             cooling_rate: float,
                             iterations: int) -> Tuple[float, float]:
    """
    Performs the Fast Simulated Annealing algorithm for a given objective function.

    :param objective: The objective function to optimize.
    :type objective: Callable[[float], float]
    :param initial_state: The initial state.
    :type initial_state: float
    :param initial_temperature: The initial temperature.
    :type initial_temperature: float
    :param cooling_rate: The cooling rate.
    :type cooling_rate: float
    :param iterations: The number of iterations to perform.
    :type iterations: int
    :return: The final state and its objective value.
    :rtype: Tuple[float, float]

    Example Usage:
    ----------------
    objective = lambda x: -x**2 + x + 10  # Simple parabolic function with maximum at x=0.5
    initial_state = 0
    initial_temperature = 10
    cooling_rate = 0.99
    iterations = 10000
    state, value = fast_simulated_annealing(objective, initial_state, initial_temperature, cooling_rate, iterations)
    print(f"Final state: {state}")
    print(f"Objective value: {value}")
    """
    current_state = initial_state
    current_temperature = initial_temperature

    for i in range(iterations):
        proposed_state = current_state + np.random.standard_cauchy()

        delta = objective(proposed_state) - objective(current_state)
        if delta > 0:
            current_state = proposed_state
        else:
            acceptance_probability = np.exp(
                delta / (current_temperature / np.log(i+2)))
            if np.random.rand() < acceptance_probability:
                current_state = proposed_state

        current_temperature *= cooling_rate

    return current_state, objective(current_state)


def simulated_quenching(objective: Callable[[float], float],
                        initial_state: float,
                        initial_temperature: float,
                        cooling_rate: float,
                        iterations: int) -> Tuple[float, float]:
    """
    Performs the Simulated Quenching algorithm for a given objective function.

    :param objective: The objective function to optimize.
    :type objective: Callable[[float], float]
    :param initial_state: The initial state.
    :type initial_state: float
    :param initial_temperature: The initial temperature.
    :type initial_temperature: float
    :param cooling_rate: The cooling rate. For quenching, this should be a large number.
    :type cooling_rate: float
    :param iterations: The number of iterations to perform.
    :type iterations: int
    :return: The final state and its objective value.
    :rtype: Tuple[float, float]

    Example Usage:
    ----------------
    objective = lambda x: -x**2 + x + 10  # Simple parabolic function with maximum at x=0.5
    initial_state = 0
    initial_temperature = 10
    cooling_rate = 0.5  # High cooling rate for simulated quenching
    iterations = 10000
    state, value = simulated_quenching(objective, initial_state, initial_temperature, cooling_rate, iterations)
    print(f"Final state: {state}")
    print(f"Objective value: {value}")
    """
    current_state = initial_state
    current_temperature = initial_temperature

    for _ in range(iterations):
        proposed_state = current_state + np.random.normal()

        if objective(proposed_state) > objective(current_state):
            current_state = proposed_state
        else:
            acceptance_probability = np.exp(
                (objective(proposed_state) - objective(current_state)) / current_temperature)

            if np.random.rand() < acceptance_probability:
                current_state = proposed_state

        current_temperature *= cooling_rate

    return current_state, objective(current_state)


def very_fast_simulated_reannealing(objective: Callable[[float], float],
                                    initial_state: float,
                                    initial_temperature: float,
                                    cooling_rate: float,
                                    iterations: int) -> Tuple[float, float]:
    """
    Performs the Very Fast Simulated Reannealing algorithm for a given objective function.

    :param objective: The objective function to optimize.
    :type objective: Callable[[float], float]
    :param initial_state: The initial state.
    :type initial_state: float
    :param initial_temperature: The initial temperature.
    :type initial_temperature: float
    :param cooling_rate: The cooling rate. For VFSR, this value may not be used as the cooling rate is adaptive.
    :type cooling_rate: float
    :param iterations: The number of iterations to perform.
    :type iterations: int
    :return: The final state and its objective value.
    :rtype: Tuple[float, float]

    Example Usage:
    ----------------
    objective = lambda x: -x**2 + x + 10  # Simple parabolic function with maximum at x=0.5
    initial_state = 0
    initial_temperature = 10
    cooling_rate = 0.99
    iterations = 10000
    state, value = very_fast_simulated_reannealing(objective, initial_state, initial_temperature, cooling_rate, iterations)
    print(f"Final state: {state}")
    print(f"Objective value: {value}")
    """
    current_state = initial_state
    best_state = initial_state
    current_temperature = initial_temperature
    best_value = objective(initial_state)

    for i in range(iterations):
        proposed_state = current_state + np.random.standard_cauchy()

        delta = objective(proposed_state) - objective(current_state)
        if delta > 0:
            current_state = proposed_state
            if objective(proposed_state) > best_value:
                best_value = objective(proposed_state)
                best_state = proposed_state
                current_temperature /= cooling_rate  # slow down cooling
        else:
            acceptance_probability = np.exp(
                delta / (current_temperature / np.log(i+2)))
            if np.random.rand() < acceptance_probability:
                current_state = proposed_state

        current_temperature *= cooling_rate  # cooling

    return best_state, best_value


def threshold_accepting(objective: Callable[[float], float],
                        initial_state: float,
                        threshold: float,
                        step_size: float,
                        iterations: int) -> Tuple[float, float]:
    """
    Performs the Threshold Accepting algorithm for a given objective function.

    :param objective: The objective function to optimize.
    :type objective: Callable[[float], float]
    :param initial_state: The initial state.
    :type initial_state: float
    :param threshold: The threshold for accepting worse solutions.
    :type threshold: float
    :param step_size: The step size for generating new candidate solutions.
    :type step_size: float
    :param iterations: The number of iterations to perform.
    :type iterations: int
    :return: The final state and its objective value.
    :rtype: Tuple[float, float]

    Example Usage:
    ----------------
    objective = lambda x: -x**2 + x + 10  # Simple parabolic function with maximum at x=0.5
    initial_state = 0
    threshold = 0.1
    step_size = 0.01
    iterations = 10000
    state, value = threshold_accepting(objective, initial_state, threshold, step_size, iterations)
    print(f"Final state: {state}")
    print(f"Objective value: {value}")
    """
    current_state = initial_state

    for _ in range(iterations):
        proposed_state = current_state + step_size * np.random.uniform(-1, 1)

        delta = objective(proposed_state) - objective(current_state)
        if delta > 0 or -delta <= threshold:
            current_state = proposed_state

    return current_state, objective(current_state)


class AnnealingOptimizer:
    def __init__(self, objective: Callable[[float], float], initial_state: float, step_size: float):
        self.objective = objective
        self.current_state = initial_state
        self.step_size = step_size
        self.best_state = initial_state
        self.best_value = objective(initial_state)

    def _acceptance_probability(self, delta: float, temperature: float) -> float:
        if delta > 0:
            return 1.0
        else:
            return np.exp(delta / temperature)

    def simulated_annealing(self, initial_temperature: float, cooling_rate: float, iterations: int):
        temperature = initial_temperature
        for _ in range(iterations):
            proposed_state = self.current_state + self.step_size * np.random.normal()
            delta = self.objective(proposed_state) - \
                self.objective(self.current_state)

            if self._acceptance_probability(delta, temperature) > np.random.rand():
                self.current_state = proposed_state
                if self.objective(self.current_state) > self.best_value:
                    self.best_value = self.objective(self.current_state)
                    self.best_state = self.current_state

            temperature *= cooling_rate

        return self.best_state, self.best_value

    def threshold_accepting(self, threshold: float, iterations: int):
        for _ in range(iterations):
            proposed_state = self.current_state + \
                self.step_size * np.random.uniform(-1, 1)
            delta = self.objective(proposed_state) - \
                self.objective(self.current_state)
            if delta > 0 or -delta <= threshold:
                self.current_state = proposed_state

        return self.current_state, self.objective(self.current_state)

# Example usage:


# Simple parabolic function with maximum at x=0.5
def objective(x): return -x**2 + x + 10


initial_state = 0
step_size = 0.01

optimizer = AnnealingOptimizer(objective, initial_state, step_size)

# Perform Simulated Annealing
initial_temperature = 10
cooling_rate = 0.99
iterations = 10000
state, value = optimizer.simulated_annealing(
    initial_temperature, cooling_rate, iterations)
print(f"Final state (SA): {state}")
print(f"Objective value (SA): {value}")

# Perform Threshold Accepting
optimizer = AnnealingOptimizer(
    objective, initial_state, step_size)  # reset the optimizer
threshold = 0.1
iterations = 10000
state, value = optimizer.threshold_accepting(threshold, iterations)
print(f"Final state (TA): {state}")
print(f"Objective value (TA): {value}")
