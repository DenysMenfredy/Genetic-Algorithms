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
        x = np.array(self.chromosome)
    
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(c * x))
    
        exp_arg1 = -b * np.sqrt(sum1 / self.dimension)
        exp_arg2 = sum2 / self.dimension
    
        # Clip to prevent overflow (exp(700) is near the limit)
        term1 = -a * np.exp(np.clip(exp_arg1, -700, 700))
        term2 = -np.exp(np.clip(exp_arg2, -700, 700))
        
        return term1 + term2 + a + np.exp(1)

class Rastrigin(Agent):
    
    higher_limit = 5.12
    lowest_limit = -5.12

    def __init__(self, dimension=None, chromosome=None):
        super().__init__(dimension=dimension, chromosome=chromosome)

    def fitness_function(self):
        return (10 * self.dimension) + sum([(x_i ** 2) - 10 * \
                np.cos((2 * np.pi) * x_i) for x_i in self.chromosome])
