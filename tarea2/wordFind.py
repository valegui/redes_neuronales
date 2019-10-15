import random
import string
import numpy as np
from GeneticAlgorithm import GeneticAlgorithm
from utils import *


def fitness_function(goal):
    def fitness(individual):
        # The number of characters that are different to the expected
        result = 0
        for i in range(len(goal)):
            if goal[i] != individual[i]:
                result += 1
        return -result
    return fitness


def gene_generator():
    return random.choice(string.ascii_letters)


def is_done(fitness_values):
    done = map(lambda f: f == 0, fitness_values)
    done = list(done)
    # Stops when there is an element that has fitness value equal to 0
    try:
        idx = done.index(True)
    except ValueError:
        idx = None
    return idx


def run(word, mr=0.1, fitness_analysis=False, show=True):
    ga = GeneticAlgorithm(population_size=100, fitness_fun=fitness_function(word),
                          gene_generator=gene_generator, gene_number=len(word),
                          mutation_rate=mr,
                          termination_cond=is_done, max_iter=200)

    best, iters = ga.run()
    print("word to find > ", word)
    print("got: ", ''.join(best))
    print(f"in {iters} iterations")
    if fitness_analysis:
        best, avg, worst = ga.get_analysis()
        fitness_per_generation(best, avg, worst, show=show, problem="Word Find", to_find=word)


if __name__ == "__main__":
    run("MEMEO", mr=0.2)
    run("holaninAS", mr=0.2)
    run("hello", mr=0.2)
    run("helloworld")
