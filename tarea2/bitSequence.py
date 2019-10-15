import numpy as np
import random
from utils import *
from GeneticAlgorithm import GeneticAlgorithm


def fitness_function(goal):
    def fitness(individual):
        # The number of bits that are different to the expected
        result = 0
        for i in range(len(goal)):
            if goal[i] != individual[i]:
                result += 1
        return -result
    return fitness


def gene_generator():
    return 1 if random.random() > .5 else 0


def is_done(fitness_values):
    done = map(lambda f: f == 0, fitness_values)
    done = list(done)
    # Stops when there is an element that has fitness value equal to 0
    try:
        idx = done.index(True)
    except ValueError:
        idx = None
    return idx


def run(seq, mr=0.1, fitness_analysis=False, show=True):
    ga = GeneticAlgorithm(population_size=100, fitness_fun=fitness_function(seq),
                          gene_generator=gene_generator, gene_number=len(seq),
                          mutation_rate=mr,
                          termination_cond=is_done, max_iter=100, seed=3822)

    best, iters = ga.run()
    print("seq to find > ", seq)
    print("got: ", best)
    print(f"in {iters} iterations")
    if fitness_analysis:
        best, avg, worst = ga.get_analysis()
        fitness_per_generation(best, avg, worst, show=show, problem="Bit Sequence",
                               to_find=''.join([str(i) for i in seq]))


def run_configurations(seq):
    populations = np.arange(50, 1001, 50)
    mutation_rates = np.arange(0, 1.1, 0.1)
    map_iters = np.empty((np.size(populations), np.size(mutation_rates)))
    map_fitness = np.empty((np.size(populations), np.size(mutation_rates)))
    for i in range(np.size(populations)):
        print(f"- Running for population size = {populations[i]}")
        for j in range(np.size(mutation_rates)):
            print(f"-- Running for mutation rate = {mutation_rates[j]}")
            ga = GeneticAlgorithm(population_size=populations[i], fitness_fun=fitness_function(seq),
                                  gene_generator=gene_generator, gene_number=len(seq),
                                  mutation_rate=mutation_rates[j],
                                  termination_cond=is_done, max_iter=60, seed=3822)
            _, iters = ga.run()
            best, _, _ = ga.get_analysis()
            map_iters[i, j] = iters
            map_fitness[i, j] = best[-1]
    heatmap_configurations(populations, mutation_rates, map_iters, type='I',
                           problem="Bit Sequence", to_find=''.join([str(i) for i in seq]))
    heatmap_configurations(populations, mutation_rates, map_fitness, type='F',
                           problem="Best fitness for Bit Sequence", to_find=''.join([str(i) for i in seq]))


if __name__ == "__main__":
    run([0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1], fitness_analysis=True, mr=0.2)
    run([0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], fitness_analysis=True, mr=0.2)
    run_configurations([0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
