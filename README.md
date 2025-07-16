# Genetic Algorithms
In this repository, I will show some implementations of **genetic algorithms** that I using in my college projects.
## What are Genetic Algorithms?
Genetic algorithms are a heuristic search method inspired in the theory of natural evolution by **Charles Darwin's**. It mimics the natural selection where the fittest individuals are selected to reproduce in order to produce offspring for the next generation.
Firstly introduced in 1975 by John Holland in his book *Adaptation in Natural and Artificial Systems*[[1]](#1). And later by David E. Goldberg in his book *Genetic algorithms in search, optimization, and machine learning*[[2]](#2).
## Where GAs are used?
  * Optimization problems
  * Search problems
  * Machine Learning
  * Reinforcement Learning
  
## Structure of a Genetic Algorithm:
  1. Randomly initialize the population with n individuals
  2. Evalute the population calculating the fitness for each individual
  3. Repeat until convergence or num of generations reached:
     - Select parents from population
     - Crossover and generate the new population
     - Mutate the new population
     - Evaluate the population
  4. Return the best individual from population
  
  **The structure above is for the simple genetic algorithm defined by David E. Goldberg**
  
## References
<a id="1">[1]</a> Holland, J.H. (1975) Adaptation in Natural and Artificial Systems. University of Michigan Press, Ann Arbor. (2nd Edition, MIT Press, 1992.).

<a id="2">[2]</a> Goldberg, D.E. (1989) Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley Longman Publishing Co. Inc., Boston, MA, USA.

## Instructions

### 1. Install dependencies

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management. To install dependencies, run:

```
uv pip install -r pyproject.toml
```

Or, if you want to use the lockfile:

```
uv pip install -r uv.lock
```

### 2. Run the Genetic Algorithm

To run the main script with default parameters:

```
uv python main.py
```

To customize parameters, pass them as CLI arguments. For example:

```
uv python main.py --size_pop 300 --generations 150 --crossover_rate 0.7 --mutation_rate 0.05 --agent_dimension 20 --enable_interactive_plot
```

**Available CLI arguments:**
- `--size_pop` (int): Population size
- `--generations` (int): Number of generations
- `--crossover_rate` (float): Crossover rate
- `--mutation_rate` (float): Mutation rate
- `--agent_dimension` (int): Agent dimension
- `--enable_interactive_plot`: Enable live plotting of fitness metrics

You can combine any of these arguments as needed. If an argument is not provided, the script will use its default value.
