import random
from GeneticAlgorithm import GeneticAlgorithm

NUMBER_OF_GENES = 5
W = 15
BOX_WEIGHT = [12, 2, 1, 1, 4]
BOX_VALUE = [4, 2, 2, 1, 10]


def fitness_function(individual):
    # The value of the bag, minus a penalty.
    assert len(individual) == len(BOX_WEIGHT)
    bag_weight = sum([a * b for a, b in zip(individual, BOX_WEIGHT)])
    bag_value = sum([a * b for a, b in zip(individual, BOX_VALUE)])
    penalty = 0
    if bag_weight > W:  # Penalty for violation of the fitness function
        penalty = sum(BOX_WEIGHT) * abs(bag_weight - W)
    return bag_value - penalty


def gene_generator():
    return 1 if random.random() > .5 else 0


def is_done(fitness_values):
    max_fitness = max(fitness_values)
    count_max = fitness_values.count(max_fitness)
    # Stops when 95% of the population has the same max fitness
    if max_fitness > 0 and count_max >= len(fitness_values) * 0.95:
        return fitness_values.index(max_fitness)
    else:
        return None


if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=100, fitness_fun=fitness_function,
                          gene_generator=gene_generator, gene_number=NUMBER_OF_GENES,
                          mutation_rate=0.2,
                          termination_cond=is_done, max_iter=100)

    best, _ = ga.run()
    print("best individual > ", best)
    print("total weight: ", sum([a * b for a, b in zip(best, BOX_WEIGHT)]))
    print("total value: ", sum([a * b for a, b in zip(best, BOX_VALUE)]))
