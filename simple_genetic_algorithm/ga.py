from random import sample, randrange, random
from simple_genetic_algorithm.agent import Agent
import numpy as np
from os import path
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import time
from IPython.display import display, clear_output
import sys

class GeneticAlgorithm:
    
    def __init__(self, **params):
        self.size_pop = params["size_pop"]
        self.num_generations = params["generations"]
        self.crossover_rate = params["crossover_rate"]
        self.mutation_rate = params["mutation_rate"]
        self.best_individual = None
        self.agent = params["agent"]
        self.agent_dimension = params["agent_dimension"]
        self.execution_name = params["execution_name"]
        self.data_path = f'data/{self.execution_name}.npy'
        
        # Interactive plotting attributes
        self.enable_interactive_plot = params.get("enable_interactive_plot", False)
        self.plot_update_interval = params.get("plot_update_interval", 100)  # milliseconds
        self.colab_mode = params.get("colab_mode", False)  # For Google Colab compatibility
        self.fig = None
        self.ax = None
        self.lines = {}
        self.current_generation = 0
        self.plot_thread = None
        self.stop_plotting = False
        self.fitness_history = {'bests': [], 'average': [], 'worsts': []}
    
    def start(self, ):
        """Main method of the algorithm"""
        
        self.resetData()
        
        # Initialize interactive plot if enabled
        if self.enable_interactive_plot:
            if self.colab_mode:
                self.initColabPlot()
            else:
                self.initInteractivePlot()
        
        population = self.initialPop()
        self.evalute(population)
        self.findBest(population)
        self.saveData(population)
        
        # Update plot for initial generation
        if self.enable_interactive_plot:
            self.updatePlot()
        
        for gen in range(1, self.num_generations):
            print(f'running generation {gen}')
            self.current_generation = gen
            population = self.reproduce(population)
            self.evalute(population)
            self.findBest(population)
            self.saveData(population)
            
            # Update plot for current generation
            if self.enable_interactive_plot:
                if self.colab_mode:
                    self.updateColabPlot()
                else:
                    self.updatePlot()
                    plt.pause(0.01)  # Small pause to allow plot update
        
        # Keep plot alive after algorithm finishes
        if self.enable_interactive_plot:
            self.stop_plotting = True
            if self.colab_mode:
                self.finalColabPlot()
            else:
                print("Algorithm finished. Close the plot window to exit.")
                plt.show()
        
        return self.best_individual
    
    def initColabPlot(self):
        """Initialize plotting for Google Colab environment"""
        # Configure matplotlib for Colab
        try:
            # Essential Colab plotting setup
            import matplotlib
            matplotlib.use('inline')  # Use inline backend for Colab
            import matplotlib.pyplot as plt
            
            # Magic command equivalent
            from IPython import get_ipython
            if get_ipython() is not None:
                get_ipython().run_line_magic('matplotlib', 'inline')
            
            # Test plot to verify setup
            plt.figure(figsize=(8, 4))
            plt.plot([0, 1], [0, 1], 'b-', label='Test')
            plt.title('Colab Plotting Test - Setup Complete')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            print("✅ Interactive plotting enabled for Google Colab")
            print("📊 Plot will update after each generation...")
            
        except Exception as e:
            print(f"❌ Error setting up Colab plotting: {e}")
            print("💡 Try adding %matplotlib inline at the top of your cell")
    
    def updateColabPlot(self):
        """Update plot in Google Colab with cell output refresh"""
        if not self.enable_interactive_plot:
            return
            
        try:
            # Read current data from file
            generations = np.arange(self.current_generation + 1)
            bests = np.ndarray((0))
            average = np.ndarray((0))
            worsts = np.ndarray((0))
            
            with open(path.abspath(self.data_path), "rb") as file:
                for _ in range(self.current_generation + 1):
                    all_fitness = np.load(file)
                    bests = np.append(bests, min(all_fitness))
                    worsts = np.append(worsts, max(all_fitness))
                    average = np.append(average, all_fitness.mean())
            
            # Store history for smoother updates
            self.fitness_history['bests'] = bests.tolist()
            self.fitness_history['average'] = average.tolist()
            self.fitness_history['worsts'] = worsts.tolist()
            
            # Update plot every few generations to avoid too much output
            if self.current_generation % 5 == 0 or self.current_generation == 1:
                clear_output(wait=True)
                
                # Create plot using plt.figure (not fig, ax)
                plt.figure(figsize=(12, 6))
                plt.plot(generations, bests, 'g-', label='Best', linewidth=2, marker='o', markersize=3)
                plt.plot(generations, average, 'b-', label='Average', linewidth=2, marker='s', markersize=3)
                plt.plot(generations, worsts, 'r-', label='Worst', linewidth=2, marker='^', markersize=3)
                
                plt.xlabel('Generation')
                plt.ylabel('Fitness Value')
                plt.title(f'Real-time Fitness Evolution - Generation {self.current_generation}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Force display - this is the key fix!
                plt.show()
                
                # Print current statistics
                print(f"📊 Generation {self.current_generation}:")
                print(f"  🟢 Best Fitness: {bests[-1]:.6f}")
                print(f"  🔵 Average Fitness: {average[-1]:.6f}")
                print(f"  🔴 Worst Fitness: {worsts[-1]:.6f}")
                
                if len(bests) > 1:
                    improvement = ((bests[0] - bests[-1]) / abs(bests[0]) * 100) if bests[0] != 0 else 0
                    print(f"  📈 Improvement: {improvement:.2f}%")
                    
                print("-" * 50)
                
        except Exception as e:
            print(f"❌ Error updating Colab plot: {e}")
            print("💡 Try restarting runtime if plots aren't appearing")
    
    def finalColabPlot(self):
        """Show final plot in Google Colab"""
        clear_output(wait=True)
        
        generations = np.arange(len(self.fitness_history['bests']))
        bests = np.array(self.fitness_history['bests'])
        average = np.array(self.fitness_history['average'])
        worsts = np.array(self.fitness_history['worsts'])
        
        plt.figure(figsize=(12, 8))
        plt.plot(generations, bests, 'g-', label='Best', linewidth=2, marker='o', markersize=4)
        plt.plot(generations, average, 'b-', label='Average', linewidth=2, marker='s', markersize=4)
        plt.plot(generations, worsts, 'r-', label='Worst', linewidth=2, marker='^', markersize=4)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Final Fitness Evolution Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("🎉 Genetic Algorithm Completed!")
        print(f"Total Generations: {len(generations)}")
        print(f"Final Best Fitness: {bests[-1]:.6f}")
        print(f"Final Average Fitness: {average[-1]:.6f}")
        print(f"Improvement: {((bests[0] - bests[-1]) / bests[0] * 100):.2f}%")
    
    def initInteractivePlot(self):
        """Initialize the interactive plot"""
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Initialize empty lines for each metric
        self.lines['bests'], = self.ax.plot([], [], 'g-', label='Best', linewidth=2)
        self.lines['average'], = self.ax.plot([], [], 'b-', label='Average', linewidth=2)
        self.lines['worsts'], = self.ax.plot([], [], 'r-', label='Worst', linewidth=2)
        
        # Set up the plot
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Fitness Value')
        self.ax.set_title('Real-time Fitness Evolution')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Set initial axis limits
        self.ax.set_xlim(0, self.num_generations)
        self.ax.set_ylim(0, 1)  # Will be updated dynamically
        
        plt.tight_layout()
    
    def updatePlot(self):
        """Update the interactive plot with current data"""
        if not self.enable_interactive_plot:
            return
            
        try:
            generations = np.arange(self.current_generation + 1)
            bests = np.ndarray((0))
            average = np.ndarray((0))
            worsts = np.ndarray((0))
            
            # Read data from file
            with open(path.abspath(self.data_path), "rb") as file:
                for _ in range(self.current_generation + 1):
                    all_fitness = np.load(file)
                    bests = np.append(bests, min(all_fitness))
                    worsts = np.append(worsts, max(all_fitness))
                    average = np.append(average, all_fitness.mean())
            
            # Update line data
            self.lines['bests'].set_data(generations, bests)
            self.lines['average'].set_data(generations, average)
            self.lines['worsts'].set_data(generations, worsts)
            
            # Update axis limits dynamically
            if len(bests) > 0:
                y_min = min(np.min(bests), np.min(average), np.min(worsts))
                y_max = max(np.max(bests), np.max(average), np.max(worsts))
                y_range = y_max - y_min
                self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            # Update x-axis if needed
            if self.current_generation > 0:
                self.ax.set_xlim(0, max(self.current_generation, 10))
            
            # Redraw the plot
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating plot: {e}")
    
    def initialPop(self, ):
        """Generate a initial population"""
        return [self.agent(dimension=self.agent_dimension) for _ in range(self.size_pop)]
    
    def evalute(self, population):
        """Evaluate each individual computing their fitness"""
        
        for individual in population:
            individual.fitness = individual.fitness_function()
            
    
    def findBest(self, population):
        """Get the best individual of the population based on its fitness"""
        
        best = sorted(population, key = lambda indv: indv.fitness)[0]
         
        if not self.best_individual:
            self.best_individual = best.copy()
            
        if best.fitness < self.best_individual.fitness:
            self.best_individual = best.copy()
            
    def reproduce(self, population):
        """Reproduce the population using th genetic operators"""
        
        mating_pool = self.selection(population)
        new_pop = self.crossover(mating_pool)
        self.mutation(new_pop)
        new_pop.sort(key = lambda indv: indv.fitness)
        percentual = int(self.size_pop * self.crossover_rate)
        percentual = percentual if percentual % 2 == 0 else percentual + 1
        population.sort(key = lambda indv: indv.fitness, reverse=True)
                
        return new_pop + population[percentual: ]
    
    def selection(self, population):
        """Select parents to mating using tournament selection method"""
        
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
        """Mating individuals to generate offspring based in crossover rate"""
        
        new_pop = []
        percentual = int(self.size_pop * self.crossover_rate)
        percentual = percentual if percentual % 2 == 0 else percentual + 1
        size = len(mating_pool)
        
        for _ in range(0, percentual, 2):
            indv1 = mating_pool[randrange(size)]
            indv2 = mating_pool[randrange(size)]
            indv12, indv21 = self.twoPointCrossover(indv1.chromosome, indv2.chromosome)
            new_pop.append(self.agent(dimension=self.agent_dimension, chromosome=indv12))
            new_pop.append(self.agent(dimension=self.agent_dimension, chromosome=indv21))
            
        return new_pop
    
    def onePointCrossover(self, chrm1, chrm2):
        """One point crossover method"""
        
        cut_point1 = randrange(len(chrm1))
        cut_point2 = cut_point1
        
        chrm12 = chrm1[ :cut_point1] + chrm2[cut_point1 :cut_point2] + chrm1[cut_point2: ]
        chrm21 = chrm2[ :cut_point1] + chrm1[cut_point1 :cut_point2] + chrm2[cut_point2: ]
        
        return(chrm12, chrm21)
    
    def twoPointCrossover(self, chrm1, chrm2):
        """Two point crossover method"""
        cut_point1 = randrange(len(chrm1))
        cut_point2 = cut_point1
        
        chrm12 = chrm1[ :cut_point1] + chrm2[cut_point1 :cut_point2] + chrm1[cut_point2: ]
        chrm21 = chrm2[ :cut_point1] + chrm1[cut_point1 :cut_point2] + chrm1[cut_point2: ]
        
        return(chrm12, chrm21)
    
    def mutation(self, population):
        """Mutate the individuals based in mutation rate"""
        
        for indiv in population:
            mutate = random() < self.mutation_rate
            if mutate:
                size = len(indiv.chromosome)
                n1, n2 = randrange(size), randrange(size)
                indiv.chromosome = indiv.chromosome[:n1] + [indiv.chromosome[n2]] + \
                indiv.chromosome[n1+1:n2] + [indiv.chromosome[n1]] + indiv.chromosome[n2+1: ]
    
    
    def saveData(self, population):
        """Save fitness data to use in graphic ploting"""
        
        all_fitness = np.array([indv.fitness for indv in population])
        with open(path.abspath(self.data_path), "ab+") as file:
            np.save(file, all_fitness)
            
    
    def plotGraphic(self, ):
        """Plot fitness vs. generation graphic (static version)"""
        
        generations = np.arange(self.num_generations)
        bests = np.ndarray((0))
        average = np.ndarray((0))
        worsts = np.ndarray((0))
        
        with open(path.abspath(self.data_path), "rb") as file:
            for _ in range(self.num_generations):
                all_fitness = np.load(file)
                bests = np.append(bests, min(all_fitness))
                worsts = np.append(worsts, max(all_fitness))
                average = np.append(average, all_fitness.mean())
                
        labels = ["bests", "average", "worsts"]
        data = [bests, average, worsts]
        
        plt.figure(figsize=(10, 6))
        for l, y in zip(labels, data):
            plt.plot(generations, y, label=l, linewidth=2)
        
        plt.title("Fitness per Generation Relationship")
        plt.xlabel("Generation")
        plt.ylabel("Fitness value")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plotInteractiveGraphic(self):
        """Enable interactive plotting mode"""
        self.enable_interactive_plot = True
        
    def resetData(self, ):
        """Reset data in file"""
        
        open(path.abspath(self.data_path), "wb").close()
