from random import uniform
import numpy as np
class Individuo:
    def __init__(self, cromossomo = None, geracao = 0):
        self.cromossomo = cromossomo if cromossomo else self.randomCromossomo()
        self.geracao = geracao
        self.fitness = 0
        
    def randomCromossomo(self, ):
        return [uniform(-512, 512) for _ in range(2)]
    
    def calcFitness(self,):
        """ EGGHOLDER FUNCTION """
        
        return -(self.cromossomo[1] + 47) * np.sin(np.sqrt(abs(self.cromossomo[1] + \
            (self.cromossomo[0] / 2) + 47))) - self.cromossomo[0] * \
                np.sin(np.sqrt(abs(self.cromossomo[0] - (self.cromossomo[1] + 47))))
    def copia(self,):
        copia = Individuo(cromossomo=self.cromossomo, geracao=self.geracao)
        copia.fitness = self.fitness
        return copia
        
    def __str__(self,):
        return f'Cromossomo: {self.cromossomo}\nGeração: {self.geracao}\nFitness: {self.fitness}\n'