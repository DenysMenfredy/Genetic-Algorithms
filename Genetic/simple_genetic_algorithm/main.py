from ag import GeneticAlgorithm


def main():
    params = {
        "size_pop": 50,
        "crossover_rate": 0.9,
        "mutation_rate": 0.05,
        "generations": 100
    }
    
    genetic_algorithm = GeneticAlgorithm(**params)
    solution = genetic_algorithm.start()
    print(solution)
    genetic_algorithm.plotGraphic()
    

if __name__ == '__main__':
    main()