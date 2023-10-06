""" **The main idea of the 8 queens algorithm is to find all possible ways to place eight queens on an 8x8 chessboard such that no two queens threaten each other. In other words, the solution should ensure that no two queens are placed in the same row, column, or diagonal on the chessboard.

The algorithm typically uses a backtracking approach to explore all possible solutions. It starts by placing a queen in the first column of the first row and then recursively tries to place a queen in each subsequent column, checking at each step whether the placement is valid according to the constraints mentioned above. If a valid placement is found, the algorithm proceeds to the next column and continues until all eight queens have been placed.

If the algorithm reaches a dead end where it is unable to place a queen in a valid position in a given column, it backtracks to the previous column and tries a different position for that queen. The algorithm continues this process until all possible solutions have been explored, at which point it returns the set of all valid solutions.

The 8 queens algorithm is a classic example of a constraint satisfaction problem and has applications in fields such as artificial intelligence, computer science, and operations research.

**The fitness function typically counts the number of pairs of queens that are attacking each other, and subtracts that number from a fixed value (usually the maximum number of pairs of queens that could possibly attack each other, which is 28). So the fitness score for a particular placement of queens is equal to (28 - number of pairs of attacking queens). A placement with a higher fitness score is considered to be a better solution than one with a lower fitness score.
fitness = - penalty 
The fitness function rule used in the 8 queens algorithm is the number of non-attacking pairs of queens that we are interested to maximize (which has the maximum value (8 choose 2)= 28 for the 8-queen’s problem).

**In the 8 queens algorithm, the single point crossover is a commonly used operator for combining genetic information from two parent solutions to create new offspring solutions. Specifically, the single point crossover operator selects a random point along each parent's chromosome (which represents a particular placement of the queens on the chessboard), and then swaps the portions of the chromosomes that come after that point.
The reason why single point crossover is a popular choice in the 8 queens algorithm is that it tends to preserve the good aspects of both parent solutions while also introducing some degree of diversity into the population of solutions. By swapping only a portion of the solution, the operator can create offspring solutions that are similar to the parents in some respects, but also have some novel variation that might help to explore new regions of the solution space.

**The 8 queens problem can be solved using various algorithms such as backtracking, genetic algorithm, and simulated annealing. In the genetic algorithm approach, the mutation operator is used to introduce new genetic material into the population. The mutation operator randomly changes the value of a gene in a chromosome with a small probability. This helps to maintain diversity in the population and avoid premature convergence to a suboptimal solution."""
import random
import matplotlib.pyplot as plt
import numpy as np
def generate_board():
    """Generate a random board"""
    board = [0] * 8
    for i in range(8):
        board[i] = random.randint(0, 7)
    return board

def fitness(board):
    """Calculate fitness of a board"""
    attacking_pairs = 0
    for i in range(8):
        for j in range(i+1, 8):
            if board[i] == board[j] or abs(board[i] - board[j]) == j - i:
                attacking_pairs += 1
    return 29 - attacking_pairs

def crossover(board1, board2):
    """Perform crossover between two boards"""
    index = random.randint(1, 6)
    child1 = board1[:index] + board2[index:]
    child2 = board2[:index] + board1[index:]
    return child1, child2

def mutate(board):
    """Mutate a board"""
    index = random.randint(0, 7)
    value = random.randint(0, 7)
    board[index] = value
    return board

def select_population(population, fitnesses, num_parents):
    """Select the parents for the next generation"""
    parents = []
    for i in range(num_parents):
        max_fitness_index = fitnesses.index(max(fitnesses))
        parents.append(population[max_fitness_index])
        population.pop(max_fitness_index)
        fitnesses.pop(max_fitness_index)
    return parents

def generate_population(population_size):
    """Generate a population of random boards"""
    population = []
    for i in range(population_size):
        population.append(generate_board())
    return population

def genetic_algorithm(population_size, num_generations):
    """Run the genetic algorithm"""
    population = generate_population(population_size)
    for i in range(num_generations):
        fitnesses = [fitness(board) for board in population]
        parents = select_population(population, fitnesses, population_size // 2)
        children = []
        for i in range(population_size // 2):
            board1 = random.choice(parents)
            board2 = random.choice(parents)
            child1, child2 = crossover(board1, board2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            children.append(child1)
            children.append(child2)
        population = parents + children
    fitnesses = [fitness(board) for board in population]
    index = fitnesses.index(max(fitnesses))
    return population[index]

solution = genetic_algorithm(population_size=100, num_generations=1000)
print("Optimal solution:", solution)



def draw_board(board):
    plt.figure(figsize=(4,4))
    plt.imshow(np.zeros((8,8)), cmap='binary')
    for i in range(8):
        for j in range(8):
            if board[i] == j:
                plt.text(j, i, '♛', fontsize=25, ha='center', va='center')
    plt.title("Optimal Solution", color = 'red', size = 15)
    plt.xticks([])
    plt.yticks([])
    plt.show()

draw_board(solution)