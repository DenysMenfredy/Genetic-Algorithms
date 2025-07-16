"""Agent base class for genetic algorithms."""
from random import uniform

class Agent:
    """Base agent for genetic algorithms, supporting chromosome and fitness logic."""
    higher_limit = None
    lowest_limit = None

    def __init__(self, dimension=None, chromosome=None):
        self.dimension = dimension
        self.chromosome = chromosome if chromosome else self.random_chromosome()
        self.fitness = 0

    def random_chromosome(self):
        """Generate a random chromosome."""
        if self.dimension is None or self.lowest_limit is None or self.higher_limit is None:
            raise ValueError("dimension, lowest_limit, and higher_limit must be set for random chromosome generation.")
        return [uniform(self.lowest_limit, self.higher_limit) for _ in range(self.dimension)]

    def fitness_function(self):
        """Agent fitness function (to be implemented by subclasses)."""
        raise NotImplementedError

    def copy(self):
        """Make a copy of the agent."""
        copy = Agent(dimension=self.dimension, chromosome=self.chromosome)
        copy.fitness = self.fitness
        return copy

    def __str__(self):
        return f'Chromosome: {self.chromosome}\nFitness: {self.fitness}\n'
    