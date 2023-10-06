# 8-Queens
# 8 Queens Algorithm

The main idea of the 8 queens algorithm is to find all possible ways to place eight queens on an 8x8 chessboard such that no two queens threaten each other. In other words, the solution should ensure that no two queens are placed in the same row, column, or diagonal on the chessboard.

## Algorithm Overview

The algorithm typically uses a backtracking approach to explore all possible solutions. It starts by placing a queen in the first column of the first row and then recursively tries to place a queen in each subsequent column, checking at each step whether the placement is valid according to the constraints mentioned above. If a valid placement is found, the algorithm proceeds to the next column and continues until all eight queens have been placed.

If the algorithm reaches a dead end where it is unable to place a queen in a valid position in a given column, it backtracks to the previous column and tries a different position for that queen. The algorithm continues this process until all possible solutions have been explored, at which point it returns the set of all valid solutions.

## Fitness Function

The fitness function used in the 8 queens algorithm counts the number of pairs of queens that are attacking each other and subtracts that number from a fixed value (usually the maximum number of pairs of queens that could possibly attack each other, which is 28). So the fitness score for a particular placement of queens is equal to (28 - number of pairs of attacking queens). A placement with a higher fitness score is considered to be a better solution than one with a lower fitness score. In other words, the fitness score is equal to the negative penalty.

## Genetic Algorithm Approach

The 8 queens problem can be solved using various algorithms such as backtracking, genetic algorithm, and simulated annealing. In the genetic algorithm approach, the single point crossover is a commonly used operator for combining genetic information from two parent solutions to create new offspring solutions. The single point crossover operator selects a random point along each parent's chromosome (which represents a particular placement of the queens on the chessboard) and then swaps the portions of the chromosomes that come after that point.

The reason why single point crossover is a popular choice in the 8 queens algorithm is that it tends to preserve the good aspects of both parent solutions while also introducing some degree of diversity into the population of solutions. By swapping only a portion of the solution, the operator can create offspring solutions that are similar to the parents in some respects but also have some novel variation that might help to explore new regions of the solution space.

## Code Implementation

The provided code demonstrates the implementation of the 8 queens algorithm using a genetic algorithm approach. It includes functions for initializing the population, calculating fitness values, selection, crossover, and mutation. The `eight_queens` function is the main function that runs the algorithm for a specified population size and maximum number of generations.

## Usage

To use the code, you can follow these steps:

1. Import the required libraries:
```python
import numpy as np
```

2. Define the necessary functions:
```python
def init_pop(pop_size):
    return np.random.randint(8, size=(pop_size, 8))

def calc_fitness(population):
    # ...

def selection(population, fitness_vals):
    # ...

def crossover(parent1, parent2, pc):
    # ...

def mutation(individual, pm):
    # ...

def crossover_mutation(selected_pop, pc, pm):
    # ...

def eight_queens(pop_size, max_generation, pc=0.7, pm=0.01):
    # ...
```

3. Set the initial population size and maximum number of generations:
```python
pop_size = 500
max_generation = 10000
```

4. Run the `eight_queens` function with the desired parameters:
```python
eight_queens(pop_size, max_generation, pc=0.7, pm=0.05)
```

This will execute the algorithm and print the best solution found.

## Conclusion

The 8 queens algorithm is a classic example of a constraint satisfaction problem and has applications in fields such as artificial intelligence, computer science, and operations research. The genetic algorithm approach provides a flexible and efficient way to solve the problem by combining the genetic information from multiple solutions to create new and potentially better solutions.
