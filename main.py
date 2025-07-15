from simple_genetic_algorithm.ga import GeneticAlgorithm
from simple_genetic_algorithm.instances import *
from utils.file import create_folder
from utils.date import generate_execution_name
from utils.graph_visualizer import GraphVisualizer


def main():
    create_folder('data') 
    execution_name = generate_execution_name() 

    colab_params = {
        "size_pop": 200,
        "generations": 100,
        "crossover_rate": 0.85,
        "mutation_rate": 0.03,
        "agent": Rastrigin,  # Your agent class
        "agent_dimension": 10,
        "execution_name": f"colab_{execution_name}",
        "enable_interactive_plot": True,
        "colab_mode": True,  # Enable Colab mode
    }
 
    # agent params
    regular_params = {
        "size_pop": 500,
        "generations": 200,
        "crossover_rate": 0.4,
        "mutation_rate": 0.3,
        "agent_dimension": 10,
        "agent": Rastrigin,
        "execution_name": execution_name,
        "enable_interactive_plot": True,  # Enable interactive plotting
        "plot_update_interval": 100  # Update interval in milliseconds
    }
    
    # Detect if running in Colab
    try:
        import google.colab
        print("Running in Google Colab")
        params = colab_params
    except ImportError:
        print("Running in regular Python environment")
        params = regular_params
    
    genetic_algorithm = GeneticAlgorithm(**params) 
    solution = genetic_algorithm.start()
    print(solution)
    

if __name__ == '__main__':
    main()
