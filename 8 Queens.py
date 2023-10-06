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
import numpy as np
def init_pop(pop_size) :
    return np.random.randint(8, size = (pop_size, 8))
initial_population = init_pop(4)
print("initial population :")
print(initial_population)
def calc_fitness(population) :
    fitness_vals = []
    for x in population :
        penalty = 0
        for i in range (8) :
            r = x[i]
            for j in range (8) :
                if i == j :
                    continue
                d = abs(i - j)
                if x[j] in [r, r-d, r+d] :
                    penalty += 1
        fitness_vals.append(penalty)
    return -1 * np.array(fitness_vals)
initial_population = init_pop(4)
print("initial population :")
print(initial_population)
fitness_vals = calc_fitness(initial_population)
print("Fitness Values :")
print(fitness_vals)
def selection(population, fitness_vals) :
    probs = fitness_vals.copy()
    probs += abs(probs.min()) + 1
    probs = probs/probs.sum()
    N = len(population)
    indices = np.arange(N)
    selected_indices = np.random.choice(indices, size=N,p=probs)
    selected_population = population[selected_indices]
    return selected_population
selected_population = selection(initial_population, fitness_vals)
print("Selected Population :")
print(selected_population)
def crossover(parent1,parent2,pc) :
    r = np.random.random()
    if r < pc :
        m = np.random.randint(1, 8)
        child1 = np.concatenate([parent1[:m], parent2[m:]])
        child2 = np.concatenate([parent2[:m], parent1[m:]])
    else :
        child1 = parent1.copy()
        child2 = parent2.copy()
    return child1, child2
def mutation(individual, pm) :
    r = np.random.random()
    if r < pm :
        m = np.random.randint(8)
        individual[m] = np.random.randint(8)
    return individual
def crossover_mutation(selected_pop,pc,pm) :
    N = len(selected_pop)
    new_pop = np.empty((N,8),dtype=int)
    for i in range(0,N,2) :
        parent1 = selected_pop[i]
        parent2 = selected_pop[i+1]
        child1, child2 = crossover(parent1, parent2,pc)
        new_pop[i] = child1
        new_pop[i+1] = child2
    for i in range(N) :
        mutation(new_pop[i],pm)
    return new_pop
def eight_queens(pop_size, max_generation,pc=0.7,pm=0.01) :
    population = init_pop(pop_size)
    best_fitness_overall = None
    for i_gen in range(max_generation) :
        fitness_vals = calc_fitness(population)
        best_i = fitness_vals.argmax()
        best_fitness = fitness_vals[best_i]
        if best_fitness_overall is None or best_fitness > best_fitness_overall :
            best_fitness_overall = best_fitness
            best_solution = population[best_i]
        print(f'\rgen = {i_gen+1:06} -f = {-best_fitness:03}', end='')
        if best_fitness == 0 :
            print('\nFound optimal solution :')
            break
        selected_pop = selection(population, fitness_vals)
        population = crossover_mutation(selected_pop,pc,pm)
    print(best_solution)
eight_queens(pop_size=500,max_generation=10000,pc=0.7,pm=0.05)
