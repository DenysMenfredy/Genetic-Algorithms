"""Problem instances for the Genetic Algorithm: HolderTable, Ackley, Rastrigin."""
import numpy as np
from simple_genetic_algorithm.agent import Agent

# pylint: disable=too-few-public-methods

class HolderTable(Agent):
    """HolderTable function instance for optimization."""
    higher_limit = 10
    lowest_limit = -10

    def fitness_function(self):
        """Compute the HolderTable fitness value for the current chromosome."""
        return -abs(
            np.sin(self.chromosome[0]) * np.cos(self.chromosome[1])
            * np.exp(abs(1 - (np.sqrt(self.chromosome[0] ** 2 + self.chromosome[1] ** 2) / np.pi)))
        )

class Ackley(Agent):
    """Ackley function instance for optimization."""
    higher_limit = 32.768
    lowest_limit = -32.768

    def fitness_function(self):
        """Compute the Ackley fitness value for the current chromosome."""
        a, b, c = 20, 0.2, 2 * np.pi
        x = np.array(self.chromosome)
        dim = self.dimension if self.dimension is not None else len(self.chromosome)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(c * x))
        exp_arg1 = -b * np.sqrt(sum1 / dim)
        exp_arg2 = sum2 / dim
        # Clip to prevent overflow (exp(700) is near the limit)
        term1 = -a * np.exp(np.clip(exp_arg1, -700, 700))
        term2 = -np.exp(np.clip(exp_arg2, -700, 700))
        return term1 + term2 + a + np.exp(1)

class Rastrigin(Agent):
    """Rastrigin function instance for optimization."""
    higher_limit = 5.12
    lowest_limit = -5.12

    def fitness_function(self):
        """Compute the Rastrigin fitness value for the current chromosome."""
        dim = self.dimension if self.dimension is not None else len(self.chromosome)
        return (
            10 * dim
            + sum(
                x_i ** 2 - 10 * np.cos(2 * np.pi * x_i)
                for x_i in self.chromosome
            )
        )
