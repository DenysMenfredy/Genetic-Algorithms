from random import sample, randrange, random
from agent import Agent
import numpy as np
from os import path
from matplotlib import pyplot as plt

class GeneticAlgorithm:
    def __init__(self, **params):
        self.size_pop = params["size_pop"]
        self.num_generations = params["generations"]
        self.crossover_rate = params["crossover_rate"]
        self.mutation_rate = params["mutation_rate"]
        self.best_individual = None
    
    
    def start(self, ):
        self.resetData()
        population =  self.initialPop()
        self.evalute(population)
        self.findBest(population)
        self.saveData(population)
        for _ in range(self.num_generations):
              population = self.reproduce(population)
              self.evalute(population)
              self.findBest(population)
              self.saveData(population)
        
        
        return self.best_individual
    
    
    def initialPop(self, ):
        return [Agent() for _ in range(self.size_pop)]
    
    def evalute(self, population):
        for individual in population:
            individual.fitness = individual.fitness_function()
            
    
    def findBest(self, population):
        best = sorted(population, key = lambda indv: indv.fitness)[0]
         
        if not self.best_individual:
            self.best_individual = best.copy()
            
        if best.fitness < self.best_individual.fitness:
            self.best_individual = best.copy()
            
    def reproduce(self, population):
        mating_pool = self.selection(population)
        new_pop = self.crossover(mating_pool)
        self.mutation(new_pop)
        new_pop.sort(key = lambda indv: indv.fitness)
        percentual = int(self.size_pop * self.crossover_rate)
        percentual = percentual if percentual % 2 == 0 else percentual + 1
        
        
        return new_pop + population[percentual: ]
    
    def selection(self, population):
        mating_pool = []
        amount = 3
        percentual = int(self.size_pop * self.crossover_rate)
        percentual = percentual if percentual % 2 == 0 else percentual + 1
        
        for _ in range(percentual):
            selecteds = sample(population, amount)
            best = sorted(selecteds ,key = lambda indv: indv.fitness)[0]
            mating_pool.append(best)
            
        return mating_pool
    
    def crossover(self, mating_pool):
        new_pop = []
        percentual = int(self.size_pop * self.crossover_rate)
        percentual = percentual if percentual % 2 == 0 else percentual + 1
        size = len(mating_pool)
        
        for _ in range(percentual):
            indv1 = mating_pool[randrange(size)]
            indv2 = mating_pool[randrange(size)]
            indv12, indv21 = self.twoPointCrossover(indv1.chromosome, indv2.chromosome)
            new_pop.append(Agent(chromosome=indv12))
            new_pop.append(Agent(chromosome=indv21))
            
        return new_pop
    
    def onePointCrossover(self, chrm1, chrm2):
        cut_point1 = randrange(len(chrm1))
        cut_point2 = cut_point1
        
        chrm12 = chrm1[ :cut_point1] + chrm2[cut_point1 :cut_point2] + chrm1[cut_point2: ]
        chrm21 = chrm2[ :cut_point1] + chrm1[cut_point1 :cut_point2] + chrm2[cut_point2: ]
        
        return(chrm12, chrm21)
    
    def twoPointCrossover(self, chrm1, chrm2):
        cut_point1 = randrange(len(chrm1))
        cut_point2 = cut_point1
        
        chrm12 = chrm1[ :cut_point1] + chrm2[cut_point1 :cut_point2] + chrm1[cut_point2: ]
        chrm21 = chrm2[ :cut_point1] + chrm1[cut_point1 :cut_point2] + chrm1[cut_point2: ]
        
        return(chrm12, chrm21)
    
    def mutation(self, population):
        for indiv in population:
            mutate = random() < self.mutation_rate
            if mutate:
                size = len(indiv.chromosome)
                n1, n2 = randrange(size), randrange(size)
                indiv.chromosome = indiv.chromosome[:n1] + [indiv.chromosome[n2]] + \
                indiv.chromosome[n1+1:n2] + [indiv.chromosome[n1]] + indiv.chromosome[n2+1: ]
    
    
    def saveData(self, population):
        all_fitness = np.array([indv.fitness for indv in population])
        with open(path.abspath('data.npy'), "ab+") as file:
            np.save(file, all_fitness)
            
    
    def plotGraphic(self, ):
        generations = np.arange(self.num_generations)
        bests = np.ndarray((0))
        average = np.ndarray((0))
        worsts = np.ndarray((0))
        
        with open(path.abspath('data.npy'), "rb") as file:
            for _ in range(self.num_generations):
                all_fitness = np.load(file)
                bests = np.append(bests, min(all_fitness))
                worsts = np.append(worsts, max(all_fitness))
                average = np.append(average, all_fitness.mean())
                
        labels = ["bests", "average", "worts"]
        data = [bests, average, worsts]
        
        for l,y in zip(labels, data):
            plt.plot(generations, y, label=l)
        
        plt.title("Fitness per Generation Relationship")
        plt.xlabel("Generation")
        plt.ylabel("Fitness value")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
        
    def resetData(self, ):
        open(path.abspath("data.npy"), "wb").close()