class KendallTests:
    def __init__(self, data):
        self.data = data
    
    def kendall_tau(self):
        concordant = 0
        discordant = 0
        
        n = len(self.data)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if self.data[i] < self.data[j]:
                    concordant += 1
                elif self.data[i] > self.data[j]:
                    discordant += 1
        
        # Compute test statistic (Kendall's tau-b)
        test_statistic = (concordant - discordant) / ((n * (n - 1)) / 2)
        
        # Compute p-value (using normal approximation)
        p_value = 2 * (1 - abs(test_statistic))  # Assuming large sample size
        
        return test_statistic, p_value
    
    def seasonal_kendall(self):
        concordant = 0
        discordant = 0
        
        n = len(self.data)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if (i + 1) % 4 == j % 4:  # Assuming seasonal period of length 1
                    if self.data[i] < self.data[j]:
                        concordant += 1
                    elif self.data[i] > self.data[j]:
                        discordant += 1
        
        # Compute test statistic
        test_statistic = (concordant - discordant) / (concordant + discordant)
        
        # Compute p-value
        p_value = 2 * min(concordant, discordant) / (n * (n - 1))
        
        return test_statistic, p_value
    
    def regional_kendall(self):
        concordant = 0
        discordant = 0
        
        n = len(self.data)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if self.data[i] < self.data[j]:
                    concordant += 1
                elif self.data[i] > self.data[j]:
                    discordant += 1
        
        # Compute test statistic
        test_statistic = (concordant - discordant) / (concordant + discordant)
        
        # Compute p-value
        p_value = 2 * min(concordant, discordant) / (n * (n - 1))
        
        return test_statistic, p_value


# Example data
data = [5, 7, 8, 3, 2, 6, 1, 4]

# Create an instance of the KendallTests class
kendall_tests = KendallTests(data)

# Perform Kendall's tau test
tau_test_statistic, tau_p_value = kendall_tests.kendall_tau()
print("Kendall's Tau Test:")
print("Test Statistic:", tau_test_statistic)
print("P-value:", tau_p_value)

# Perform Seasonal Kendall test
seasonal_test_statistic, seasonal_p_value = kendall_tests.seasonal_kendall()
print("\nSeasonal Kendall Test:")
print("Test Statistic:", seasonal_test_statistic)
print("P-value:", seasonal_p_value)

# Perform Regional Kendall test
regional_test_statistic, regional_p_value = kendall_tests.regional_kendall()
print("\nRegional Kendall Test:")
print("Test Statistic:", regional_test_statistic)
print("P-value:", regional_p_value)
