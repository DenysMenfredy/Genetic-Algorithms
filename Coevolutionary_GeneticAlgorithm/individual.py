"""Individual class for coevolutionary genetic algorithms."""
from random import uniform
import numpy as np

class Individual:
    """Represents an individual in the population for the EggHolder function."""
    def __init__(self, cromossomo=None, geracao=0):
        """Initialize an individual with a chromosome and generation."""
        self.cromossomo = cromossomo if cromossomo else self.random_cromossomo()
        self.geracao = geracao
        self.fitness = 0

    def random_cromossomo(self):
        """Generate a random chromosome for the EggHolder function."""
        return [uniform(-512, 512) for _ in range(2)]

    def calc_fitness(self):
        """Calculate the EggHolder fitness value for this individual."""
        return -(
            (self.cromossomo[1] + 47) * np.sin(np.sqrt(abs(self.cromossomo[1] + (self.cromossomo[0] / 2) + 47)))
            + self.cromossomo[0] * np.sin(np.sqrt(abs(self.cromossomo[0] - (self.cromossomo[1] + 47))))
        )

    def copia(self):
        """Return a copy of this individual."""
        copia = Individual(cromossomo=self.cromossomo, geracao=self.geracao)
        copia.fitness = self.fitness
        return copia

    def __str__(self):
        """String representation of the individual."""
        return f'Cromossomo: {self.cromossomo}\nGeração: {self.geracao}\nFitness: {self.fitness}\n'