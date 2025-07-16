"""Genetic Algorithm core implementation and live fitness visualization."""
import os
import random
import numpy as np
from utils.graph_visualizer import GraphVisualizer

class GeneticAlgorithm:
    """A simple Genetic Algorithm with live fitness plotting support."""
    # pylint: disable=too-many-instance-attributes
    def __init__(self, **params):
        """Initialize the Genetic Algorithm with parameters."""
        self.size_pop = params["size_pop"]
        self.num_generations = params["generations"]
        self.crossover_rate = params["crossover_rate"]
        self.mutation_rate = params["mutation_rate"]
        self.best_individual = None
        self.agent = params["agent"]
        self.agent_dimension = params["agent_dimension"]
        self.execution_name = params["execution_name"]
        self.data_path = f'data/{self.execution_name}.npy'
        self.enable_interactive_plot = params.get("enable_interactive_plot", False)
        self.visualizer = None
        self.current_generation = 0
        if self.enable_interactive_plot:
            self.visualizer = GraphVisualizer(self.num_generations)

    def start(self):
        """Main method of the algorithm."""
        self.reset_data()
        population = self.initial_pop()
        self.evalute(population)
        self.find_best(population)
        self.save_data(population)
        if self.visualizer is not None:
            self.visualizer.update(0, np.array([indv.fitness for indv in population]))
        for gen in range(1, self.num_generations):
            print(f'running generation {gen}')
            self.current_generation = gen
            population = self.reproduce(population)
            self.evalute(population)
            self.find_best(population)
            self.save_data(population)
            if self.visualizer is not None:
                self.visualizer.update(gen, np.array([indv.fitness for indv in population]))
        if self.visualizer is not None:
            self.visualizer.close()
        return self.best_individual

    def initial_pop(self):
        """Generate an initial population."""
        return [self.agent(dimension=self.agent_dimension) for _ in range(self.size_pop)]

    def evalute(self, population):
        """Evaluate each individual computing their fitness."""
        for individual in population:
            individual.fitness = individual.fitness_function()

    def find_best(self, population):
        """Get the best individual of the population based on its fitness."""
        best = sorted(population, key=lambda indv: indv.fitness)[0]
        if not self.best_individual:
            self.best_individual = best.copy()
        if best.fitness < self.best_individual.fitness:
            self.best_individual = best.copy()

    def reproduce(self, population):
        """Reproduce the population using the genetic operators."""
        mating_pool = self.selection(population)
        new_pop = self.crossover(mating_pool)
        self.mutation(new_pop)
        new_pop.sort(key=lambda indv: indv.fitness)
        percentual = int(self.size_pop * self.crossover_rate)
        percentual = percentual if percentual % 2 == 0 else percentual + 1
        population.sort(key=lambda indv: indv.fitness, reverse=True)
        return new_pop + population[percentual:]

    def selection(self, population):
        """Select parents to mate using tournament selection method."""
        mating_pool = []
        amount = 3
        percentual = int(self.size_pop * self.crossover_rate)
        percentual = percentual if percentual % 2 == 0 else percentual + 1
        for _ in range(percentual):
            selecteds = np.random.choice(population, amount, replace=False)
            best = sorted(selecteds, key=lambda indv: indv.fitness)[0]
            mating_pool.append(best)
        return mating_pool

    def crossover(self, mating_pool):
        """Mate individuals to generate offspring based on crossover rate."""
        new_pop = []
        percentual = int(self.size_pop * self.crossover_rate)
        percentual = percentual if percentual % 2 == 0 else percentual + 1
        size = len(mating_pool)
        for _ in range(0, percentual, 2):
            indv1 = mating_pool[np.random.randint(size)]
            indv2 = mating_pool[np.random.randint(size)]
            indv12, indv21 = self.two_point_crossover(indv1.chromosome, indv2.chromosome)
            new_pop.append(self.agent(dimension=self.agent_dimension, chromosome=indv12))
            new_pop.append(self.agent(dimension=self.agent_dimension, chromosome=indv21))
        return new_pop

    def one_point_crossover(self, chrm1, chrm2):
        """One point crossover method."""
        cut_point1 = np.random.randint(len(chrm1))
        cut_point2 = cut_point1
        chrm12 = chrm1[:cut_point1] + chrm2[cut_point1:cut_point2] + chrm1[cut_point2:]
        chrm21 = chrm2[:cut_point1] + chrm1[cut_point1:cut_point2] + chrm2[cut_point2:]
        return (chrm12, chrm21)

    def two_point_crossover(self, chrm1, chrm2):
        """Two point crossover method."""
        cut_point1 = np.random.randint(len(chrm1))
        cut_point2 = cut_point1
        chrm12 = chrm1[:cut_point1] + chrm2[cut_point1:cut_point2] + chrm1[cut_point2:]
        chrm21 = chrm2[:cut_point1] + chrm1[cut_point1:cut_point2] + chrm1[cut_point2:]
        return (chrm12, chrm21)

    def mutation(self, population):
        """Mutate the individuals based on mutation rate."""
        for indiv in population:
            mutate = random.random() < self.mutation_rate
            if mutate:
                size = len(indiv.chromosome)
                n1, n2 = np.random.randint(size), np.random.randint(size)
                indiv.chromosome = (
                    indiv.chromosome[:n1] + [indiv.chromosome[n2]] +
                    indiv.chromosome[n1+1:n2] + [indiv.chromosome[n1]] + indiv.chromosome[n2+1:]
                )

    def save_data(self, population):
        """Save fitness data to use in graphic plotting."""
        all_fitness = np.array([indv.fitness for indv in population])
        with open(os.path.abspath(self.data_path), "ab+") as file:
            np.save(file, all_fitness)

    def reset_data(self):
        """Reset data in file."""
        with open(os.path.abspath(self.data_path), "wb"):
            pass

    def plot_fitness_history(self):
        """Plot static fitness history from the .npy file for this GA run."""
        GraphVisualizer.plot_fitness_history(self.data_path)

# pylint: disable=too-few-public-methods
