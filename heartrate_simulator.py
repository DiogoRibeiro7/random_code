import random
import math
import logging

class HeartbeatSimulator:
    """
    HeartbeatSimulator class for simulating heart rates using different probability distributions.
    """

    def __init__(self, distribution: str, *args):
        """
        Initialize the HeartbeatSimulator.

        Parameters:
        - distribution (str): The type of distribution to use for heart rate simulation.
        - args: The parameters required for the chosen distribution.
        """
        self.distribution = distribution
        self.params = args
        self._validate_params()

    def _validate_params(self):
        """
        Validate the parameters based on the chosen distribution.
        """
        if self.distribution == 'normal' and len(self.params) != 2:
            raise ValueError("For normal distribution, expected 2 parameters: mean and standard deviation")
        elif self.distribution == 'gamma' and len(self.params) != 2:
            raise ValueError("For gamma distribution, expected 2 parameters: shape and scale")
        elif self.distribution == 'log_normal' and len(self.params) != 2:
            raise ValueError("For log-normal distribution, expected 2 parameters: mu and sigma")
        elif self.distribution == 'weibull' and len(self.params) != 2:
            raise ValueError("For Weibull distribution, expected 2 parameters: shape and scale")
        elif self.distribution == 'exponential' and len(self.params) != 1:
            raise ValueError("For exponential distribution, expected 1 parameter: lambda")

    def simulate_heart_rate(self) -> float:
        """
        Simulate a heart rate value based on the chosen distribution.

        Returns:
        - heart_rate (float): The simulated heart rate value.
        """
        try:
            if self.distribution == 'normal':
                mean, std_deviation = self.params
                heart_rate = random.normalvariate(mean, std_deviation)
            elif self.distribution == 'gamma':
                shape, scale = self.params
                heart_rate = random.gammavariate(shape, scale)
            elif self.distribution == 'log_normal':
                mu, sigma = self.params
                log_heart_rate = random.normalvariate(mu, sigma)
                heart_rate = math.exp(log_heart_rate)
            elif self.distribution == 'weibull':
                shape, scale = self.params
                heart_rate = random.weibullvariate(shape, scale)
            elif self.distribution == 'exponential':
                lambd = self.params[0]
                heart_rate = random.expovariate(lambd)
            else:
                raise ValueError("Invalid distribution option")

        except Exception as e:
            logging.error(f"Error occurred while simulating heart rate: {str(e)}")
            raise e

        return heart_rate

    def generate_heart_rates(self, n: int) -> list[float]:
        """
        Generate a list of heart rate values using the chosen distribution.

        Parameters:
        - n (int): The number of heart rate values to generate.

        Returns:
        - heart_rates (list): List of simulated heart rate values.
        """
        heart_rates = []
        for _ in range(n):
            heart_rate = self.simulate_heart_rate()
            while heart_rate < 40:  # Ensure heart rates are realistic (above 40)
                heart_rate = self.simulate_heart_rate()
                print(heart_rate)
            heart_rates.append(heart_rate)
        return heart_rates




simulator_normal = HeartbeatSimulator('normal', 70, 10)  # Normal distribution with mean=70, std_deviation=10
heart_rates_normal = simulator_normal.generate_heart_rates(10)  # Generate 10 heart rate values

simulator_gamma = HeartbeatSimulator('gamma', 25, 5)  # Gamma distribution with shape=2, scale=5
heart_rates_gamma = simulator_gamma.generate_heart_rates(10)  # Generate 10 heart rate values

# simulator_log_normal = HeartbeatSimulator('log_normal', 1.0, 0.5)  # Log-normal distribution with mu=1.0, sigma=0.5
# heart_rates_log_normal = simulator_log_normal.generate_heart_rates(10)  # Generate 10 heart rate values

# simulator_weibull = HeartbeatSimulator('weibull', 2, 2)  # Weibull distribution with shape=2, scale=2
# heart_rates_weibull = simulator_weibull.generate_heart_rates(10)  # Generate 10 heart rate values

# simulator_exponential = HeartbeatSimulator('exponential', 0.2)  # Exponential distribution with lambda=0.2
# heart_rates_exponential = simulator_exponential.generate_heart_rates(10)  # Generate 10 heart rate values

# Print the generated heart rate values for each distribution
print("Generated Heart Rates (Normal Distribution):")
for heart_rate in heart_rates_normal:
    print(heart_rate)

print("\nGenerated Heart Rates (Gamma Distribution):")
for heart_rate in heart_rates_gamma:
    print(heart_rate)

print("\nGenerated Heart Rates (Log-normal Distribution):")
for heart_rate in heart_rates_log_normal:
    print(heart_rate)

print("\nGenerated Heart Rates (Weibull Distribution):")
for heart_rate in heart_rates_weibull:
    print(heart_rate)

print("\nGenerated Heart Rates (Exponential Distribution):")
for heart_rate in heart_rates_exponential:
    print(heart_rate)

