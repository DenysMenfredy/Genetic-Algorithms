from agent import Agent
import numpy as np

class HolderTable(Agent):
    
    def __init__(self, chromosome=None):
        self.higher_limit = 10
        self.lowest_limit = -10
        self.size_chromosome = 2
        super().__init__(chromosome)
        

    def fitness_function(self, ):
        return - abs(np.sin(self.chromosome[0]) * np.cos(self.chromosome[1])\
               * np.exp(abs(1- ((np.sqrt(self.chromosome[0] ** 2 \
               + self.chromosome[1] ** 2)) / np.pi))))

