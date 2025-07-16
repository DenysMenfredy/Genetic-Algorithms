"""Main entry point for running the Genetic Algorithm with CLI parameters."""
import argparse
import sys
import os
from simple_genetic_algorithm.utils.file import create_folder
from simple_genetic_algorithm.utils.date import generate_execution_name
from simple_genetic_algorithm.ga import GeneticAlgorithm
from simple_genetic_algorithm import instances


def main():
    """Parse CLI arguments, configure and run the Genetic Algorithm."""
    parser = argparse.ArgumentParser(
        description="Run Genetic Algorithm with optional CLI parameters."
    )
    parser.add_argument('--size_pop', type=int, default=100, help='Population size')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--crossover_rate', type=float, default=0.9, help='Crossover rate')
    parser.add_argument('--mutation_rate', type=float, default=0.05, help='Mutation rate')
    parser.add_argument('--agent_dimension', type=int, default=10, help='Agent dimension')
    parser.add_argument(
        '--enable_interactive_plot', action='store_true',
        help='Enable interactive plotting'
    )
    args = parser.parse_args()

    create_folder('data')
    execution_name = generate_execution_name()

    params = {
        "size_pop": args.size_pop,
        "generations": args.generations,
        "crossover_rate": args.crossover_rate,
        "mutation_rate": args.mutation_rate,
        "agent_dimension": args.agent_dimension,
        "agent": instances.Rastrigin,
        "execution_name": execution_name,
        "enable_interactive_plot": args.enable_interactive_plot,
        "plot_update_interval": 100
    }

    print("Using parameters:")
    for k, v in params.items():
        if k == "agent":
            print(f"  {k}: {v.__name__}")
        else:
            print(f"  {k}: {v}")

    genetic_algorithm = GeneticAlgorithm(**params)
    solution = genetic_algorithm.start()
    print(solution)
    genetic_algorithm.plot_fitness_history()


if __name__ == '__main__':
    main()
