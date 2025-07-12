from simple_genetic_algorithm.ga import GeneticAlgorithm
from simple_genetic_algorithm.instances import *
from utils.file import create_folder
from utils.date import generate_execution_name


def main():
    create_folder('data') 
    execution_name = generate_execution_name() 

    colab_params = {
        "size_pop": 50,
        "generations": 100,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "agent": Rastrigin,  # Your agent class
        "agent_dimension": 10,
        "execution_name": "colab_test_run",
        "enable_interactive_plot": True,
        "colab_mode": True,  # Enable Colab mode
    }
 
    # agent params
    regular_params = {
        "size_pop": 100,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "generations": 100,
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
    if not genetic_algorithm.enable_interactive_plot:
        genetic_algorithm.plotGraphic()
    

if __name__ == '__main__':
    main()
