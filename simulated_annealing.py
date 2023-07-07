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
            acceptance_probability = np.exp((objective(proposed_state) - objective(current_state)) / current_temperature)
            
            if np.random.rand() < acceptance_probability:
                current_state = proposed_state

        current_temperature *= cooling_rate

    return current_state, objective(current_state)



import numpy as np
from typing import Callable

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
            acceptance_probability = np.exp(delta / (current_temperature / np.log(i+2)))
            if np.random.rand() < acceptance_probability:
                current_state = proposed_state

        current_temperature *= cooling_rate

    return current_state, objective(current_state)


import numpy as np
from typing import Callable, Tuple

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
            acceptance_probability = np.exp((objective(proposed_state) - objective(current_state)) / current_temperature)
            
            if np.random.rand() < acceptance_probability:
                current_state = proposed_state

        current_temperature *= cooling_rate

    return current_state, objective(current_state)


import numpy as np
from typing import Callable, Tuple

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
            acceptance_probability = np.exp(delta / (current_temperature / np.log(i+2)))
            if np.random.rand() < acceptance_probability:
                current_state = proposed_state

        current_temperature *= cooling_rate  # cooling

    return best_state, best_value
