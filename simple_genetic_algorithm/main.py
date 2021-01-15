from ag import GeneticAlgorithm
from instances import HolderTable


def main():
    
    # agent params
    params = {
        "size_pop": 100,
        "crossover_rate": 0.9,
        "mutation_rate": 0.03,
        "generations": 100,
        "agent_dimension": 2,
        "agent": HolderTable
    }
    
    genetic_algorithm = GeneticAlgorithm(**params) 
    solution = genetic_algorithm.start()
    print(solution)
    genetic_algorithm.plotGraphic()
    

if __name__ == '__main__':
    main()