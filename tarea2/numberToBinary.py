import numpy as np
import random
from GeneticAlgorithm import GeneticAlgorithm

NUMBER_TO_CONVERT = 121
NUMBER_OF_GENES = 8


def fitness_function(goal):
    def fitness(individual):
        # The distance to the number
        result = 0
        exp = 1
        for v in individual[::-1]:
            result += int(v) * exp
            exp *= 2
        return - abs(goal - result)
    return fitness


def gene_generator():
    return '1' if random.random() > .5 else '0'


def is_done(fitness_values):
    done = map(lambda f: f == 0, fitness_values)
    done = list(done)
    # Stops when there is an element that has fitness value equal to 0
    try:
        idx = done.index(True)
    except ValueError:
        idx = None
    return idx


if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=100, fitness_fun=fitness_function(NUMBER_TO_CONVERT),
                          gene_generator=gene_generator, gene_number=NUMBER_OF_GENES,
                          mutation_rate=0.2, termination_cond=is_done, max_iter=30)

    best, _ = ga.run()
    print(''.join(best))
