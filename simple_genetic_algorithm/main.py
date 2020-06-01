from ag import GeneticAlgorithm
from instances import HolderTable

def main():
    
    # agent params
    params = {
        "size_pop": 150,
        "crossover_rate": 0.9,
        "mutation_rate": 0.05,
        "generations": 200,
        "agent": HolderTable
    }
    
    genetic_algorithm = GeneticAlgorithm(**params) 
    solution = genetic_algorithm.start()
    print(solution)
    genetic_algorithm.plotGraphic()
    

if __name__ == '__main__':
    main()