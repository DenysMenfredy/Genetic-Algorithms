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
