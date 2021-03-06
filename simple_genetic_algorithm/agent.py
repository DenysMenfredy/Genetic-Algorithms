from random import uniform
import numpy as np

class Agent(object):
    
    # dimension = None
    higher_limit = None
    lowest_limit = None
    
    def __init__(self, dimension=None, chromosome=None):
        self.dimension = dimension
        self.chromosome = chromosome if chromosome else self.randomChromosome()
        self.fitness = 0
        
    def randomChromosome(self, ):
        """Generate a random chromosome"""
        
        return [uniform(self.lowest_limit, self.higher_limit) for _ in range(self.dimension)]
    
    def fitness_function(self, ):
        """Agent fitness function"""
        
        raise NotImplementedError
    
    def copy(self, ):
        """Make a copy of the agent"""
        
        copy = Agent(dimension=self.dimension, chromosome=self.chromosome)
        copy.fitness = self.fitness
        
        return copy
    
    def __str__(self, ):
        return f'Chromosome: {self.chromosome}\nFitness: {self.fitness}\n'
    