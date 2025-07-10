from simple_genetic_algorithm.agent import Agent
import numpy as np

class HolderTable(Agent):
    
    higher_limit = 10
    lowest_limit = -10
    
    def __init__(self, dimension=None, chromosome=None):
        # self.dimension = dimension
        super().__init__(dimension, chromosome)    

    def fitness_function(self, ):
        return - abs(np.sin(self.chromosome[0]) * np.cos(self.chromosome[1])\
               * np.exp(abs(1- ((np.sqrt(self.chromosome[0] ** 2 \
               + self.chromosome[1] ** 2)) / np.pi))))

class Ackley(Agent):
    
    higher_limit = 32.768
    lowest_limit = -32.768
    
    def __init__(self, dimension=None, chromosome=None):
        super().__init__(dimension, chromosome)

    def fitness_function(self):
        a, b, c = 20, 0.2, 2 * np.pi

        sum1 = sum([x_i** 2 for x_i in self.chromosome])
        sum2 = sum([np.cos(c * x_i) for x_i in self.chromosome])

        return -a * np.exp(-b * np.sqrt((1/self.dimension) * sum1)) - \
                np.exp((1/self.dimension)* sum2) + a + np.exp(1)

class Rastrigin(Agent):
    
    higher_limit = 5.12
    lowest_limit = -5.12

    def __init__(self, dimension=None, chromosome=None):
        super().__init__(dimension=dimension, chromosome=chromosome)

    def fitness_function(self):
        return (10 * self.dimension) + sum([(x_i ** 2) - 10 * \
                np.cos((2 * np.pi) * x_i) for x_i in self.chromosome])
