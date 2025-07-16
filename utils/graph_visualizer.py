import matplotlib.pyplot as plt
import numpy as np

class GraphVisualizer:
    def __init__(self, num_generations):
        self.num_generations = num_generations
        self.generations = []
        self.bests = []
        self.averages = []
        self.worsts = []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.best_line, = self.ax.plot([], [], 'g-', label='Best', linewidth=2)
        self.avg_line, = self.ax.plot([], [], 'b-', label='Average', linewidth=2)
        self.worst_line, = self.ax.plot([], [], 'r-', label='Worst', linewidth=2)
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Fitness Value')
        self.ax.set_title('Fitness Evolution per Generation')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        plt.ion()
        plt.show(block=False)

    def update(self, generation, fitness_arr):
        self.generations.append(generation)
        self.bests.append(np.min(fitness_arr))
        self.averages.append(np.mean(fitness_arr))
        self.worsts.append(np.max(fitness_arr))
        self.best_line.set_data(self.generations, self.bests)
        self.avg_line.set_data(self.generations, self.averages)
        self.worst_line.set_data(self.generations, self.worsts)
        self.ax.set_xlim(0, max(self.num_generations, len(self.generations)))
        y_min = min(self.bests + self.averages + self.worsts)
        y_max = max(self.bests + self.averages + self.worsts)
        y_range = y_max - y_min if y_max > y_min else 1
        self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.show() 

    @staticmethod
    def plot_fitness_history(npy_file_path):
        """Plot static fitness history (best, average, worst) from a .npy file."""
        bests, avgs, worsts = [], [], []
        try:
            with open(npy_file_path, "rb") as f:
                gen = 0
                while True:
                    try:
                        arr = np.load(f)
                        bests.append(np.min(arr))
                        avgs.append(np.mean(arr))
                        worsts.append(np.max(arr))
                        gen += 1
                    except (ValueError, EOFError):
                        break
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        generations = np.arange(len(bests))
        plt.figure(figsize=(10, 6))
        plt.plot(generations, bests, 'g-', label='Best', linewidth=2)
        plt.plot(generations, avgs, 'b-', label='Average', linewidth=2)
        plt.plot(generations, worsts, 'r-', label='Worst', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Fitness Evolution per Generation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show() 