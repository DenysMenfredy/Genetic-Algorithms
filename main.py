from simple_genetic_algorithm.ga import GeneticAlgorithm
from simple_genetic_algorithm.instances import *
from utils.file import create_folder
from utils.date import generate_execution_name


def main():
    create_folder('data') 
    execution_name = generate_execution_name() 

    # agent params
    params = {
        "size_pop": 100,
        "crossover_rate": 0.9,
        "mutation_rate": 0.03,
        "generations": 100,
        "agent_dimension": 10,
        "agent": Rastrigin,
        "execution_name": execution_name
    }
    
    genetic_algorithm = GeneticAlgorithm(**params) 
    solution = genetic_algorithm.start()
    print(solution)
    genetic_algorithm.plotGraphic()
    

if __name__ == '__main__':
    main()
