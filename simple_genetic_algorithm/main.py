from ga import GeneticAlgorithm
from instances import *


def main():
    
    # agent params
    params = {
        "size_pop": 100,
        "crossover_rate": 0.9,
        "mutation_rate": 0.03,
        "generations": 100,
        "agent_dimension": 10,
        "agent": Rastrigin
    }
    
    genetic_algorithm = GeneticAlgorithm(**params) 
    solution = genetic_algorithm.start()
    print(solution)
    genetic_algorithm.plotGraphic()
    

if __name__ == '__main__':
    main()