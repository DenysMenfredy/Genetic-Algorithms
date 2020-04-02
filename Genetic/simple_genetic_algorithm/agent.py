from random import uniform
import numpy as np

class Agent:
    def __init__(self, chromosome=None):
        self.size_chromosome = 2
        self.chromosome = chromosome if chromosome else self.randomChromosome()
        self.fitness = 0
        
        
    def randomChromosome(self, ):
        return [uniform(-10, 10) for _ in range(self.size_chromosome)]
    
    def fitness_function(self, ):
        """Holder Table Function"""
        
        return -abs(np.sin(self.chromosome[0]) * np.cos(self.chromosome[1])\
             * np.exp(abs(1- ((np.sqrt(self.chromosome[0] ** 2 + self.chromosome[1] ** 2)) / np.pi))))
    
    def copy(self, ):
        copy = Agent(chromosome=self.chromosome)
        copy.fitness = self.fitness
        return copy
    
    def __str__(self, ):
        return f'Chromosome: {self.chromosome}\nFitness: {self.fitness}\n'
    