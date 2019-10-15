import random


class GeneticAlgorithm:
    """
    GeneticAlgorithm class
    """
    def __init__(self, population_size, fitness_fun, gene_generator, gene_number,
                 mutation_rate, termination_cond, max_iter, seed=None):
        """
        Genetic Algorithm class constructor
        :param population_size: int. Number of individuals in the population.
        :param fitness_fun: function. Function that receives an individual an return its fitness.
        :param gene_generator: function. Function that generates a valid gene for an individual.
        :param gene_number: int. Number of genes per individual.
        :param mutation_rate: float. Percentage of genes to mutate in an individual.
        :param termination_cond: function. Function that receives the fitness values of the generation
            and returns the index of the best individual if the fitness matches the "termination" criteria,
            or None if there must be another generation.
        :param max_iter: int. Number of max. iterations/generations.
        :param seed: int or None. Seed for random.
        """
        self.population_size = population_size
        self.fitness_fun = fitness_fun
        self.gene_generator = gene_generator
        self.gene_number = gene_number
        self.mutation_rate = mutation_rate
        self.termination_cond = termination_cond
        self.max_iter = max_iter
        # Placeholders for the population and its fitness values
        self.population = []
        self.fitness_values = []
        # For analysis
        self.best_fitness = []
        self.avg_fitness = []
        self.worst_fitness = []
        if seed is not None:
            random.seed(seed)

    def run(self):
        """
        Run the genetic algorithm for 'max_iter' iterations.
        If a certain individual meets the termination condition, the algorithm stops before those iterations.
        :return: The individual with best fitness
                 The number of iterations completed when the algorithm stopped
        """
        self.init_population()
        for i in range(self.max_iter):
            self.evaluate_fitness()
            self.add_to_analysis()  # For analysis
            terminated = self.termination_cond(self.fitness_values)
            if terminated is not None:
                return self.population[terminated], i + 1
            parents = self.select_mating_pool()  # Selection of parents
            offspring_crossover = self.crossover(parents)  # Crossover of parents to get offspring
            offspring_mutation = self.mutation(offspring_crossover)  # Mutation of the offspring
            self.population = offspring_mutation  # The population with new
        self.evaluate_fitness()
        return self.population[self.fitness_values.index(max(self.fitness_values))], self.max_iter

    def init_population(self):
        """
        Creates the population of the size 'population_size', in which each individual has 'gene_number'.
        :return:
        """
        for i in range(self.population_size):
            self.population.append([self.gene_generator() for _ in range(self.gene_number)])
        assert len(self.population) == self.population_size

    def evaluate_fitness(self):
        """
        Computes the fitness function for each individual of the population.
        :return:
        """
        self.fitness_values = []
        for p in self.population:
            self.fitness_values.append(self.fitness_fun(p))
        assert len(self.fitness_values) == self.population_size

    def select_mating_pool(self):
        """
        Creates a mating pool
        :return: Array with parents for the next generation
        """
        mating_pool = []
        for i in range(self.population_size * 2):  # Append parents, 2 times the size of the population
            mating_pool.append(self.select_parent())
        assert len(mating_pool) == 2 * self.population_size
        return mating_pool

    def select_parent(self):
        """
        Selects an individual of the population to be a parent with the tournament selection algorithm.
        :return: An individual of the population. The one that won the tourney.
        """
        k = 5
        parents = random.sample(range(self.population_size), k)  # Gets k random index of individuals
        best = parents[0]
        for i in parents:
            if self.fitness_values[i] > self.fitness_values[best]:  # Saves the index if the fitness is better
                best = i
        return self.population[best]  # Returns the best individual between the k selected

    def crossover(self, parents):
        """
        Given a set of parents, selects a random crossover point between 1/4 and 3/4 the length of an individual
        and every two parent produces a new individual.
        :param parents: Array with individuals that will generate the offspring.
        :return: Offspring, the array that will be the next population.
        """
        assert len(parents) == 2 * self.population_size
        offspring = []
        crossover_point = random.randint(int(self.gene_number / 4), int(3 * self.gene_number / 4))
        for i in range(self.population_size):
            p1 = parents.pop()
            p2 = parents.pop()
            offspring.append(p1[:crossover_point] + p2[crossover_point:])
        assert len(offspring) == self.population_size
        return offspring

    def mutation(self, offspring):
        """
        Given an offspring array, calculates the number of genes to mutate in an individual.
        For each individual in the offspring changes the number of genes in random positions.
        :param offspring: Array of individuals.
        :return: Offspring with mutated genes.
        """
        genes_to_mutate = int(self.mutation_rate * self.gene_number)
        for o in offspring:  # For each individual in the offspring
            genes = random.sample(range(self.gene_number), genes_to_mutate)  # Indexes of genes to mutate
            for idx in genes:  # For each gene to mutate...
                new_gene = self.gene_generator()  # ...generates a new gene...
                while new_gene == o[idx]:  # ...until the gene is different...
                    new_gene = self.gene_generator()
                o[idx] = new_gene  # ...and replaces the original gene.
        return offspring

    def get_analysis(self):
        """
        Return the values of the best fitness, the average fitness and the worst fitness per generation.
        Each return value is an array.
        :return: Best fitness array, average fitness array, worst fitness array
        """
        return self.best_fitness, self.avg_fitness, self.worst_fitness

    def add_to_analysis(self):
        """
        Add values of best fitness, average fitness and worst fitness to the corresponding array.
        :return:
        """
        self.best_fitness.append(max(self.fitness_values))
        self.avg_fitness.append(sum(self.fitness_values) / len(self.fitness_values))
        self.worst_fitness.append(min(self.fitness_values))
