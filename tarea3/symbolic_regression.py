from GeneticAlgorithm import GeneticAlgorithm
from AST import AST
from arboles import *
from utils import fitness_per_generation, heatmap_configurations
import math
import sys
import numpy as np

TERMINAL = list(range(-10, 11, 1)) + ['x'] * 21
AST1 = AST([AddNode, SubNode, MultNode], TERMINAL, 0.5)
AST2 = AST([AddNode, SubNode, MultNode, DivNode], TERMINAL, 0.5)

NEG_INF = float("-inf")
MIN_SYS = - sys.maxsize - 1
values_x = range(-100, 101, 1)


# ecuacion a encontrar
def eqtn(x):
    return x * x + x - 6


# Funcion de fitness que calcula la diferencia para todos los valores de x definidos
# Si se tiene un valor infinito o nan entonces se retorna el minimo definido (castigo)
# Esto ultimo no afecta a los valores cuando no se usa el nodo division (no debe haber un inf en ese caso).
# Se agrega penalizacion por el alto del arbol.
def fitness(individual):
    diff = 0
    for x in values_x:
        diff += - abs(eqtn(x) - individual.eval(dict={'x': x}))
    # penalizacion por uso de division por cero
    if math.isnan(diff) or math.isinf(diff) or math.isinf(-diff):
        return MIN_SYS  # - math.inf
    # penalizacion por alto del arbol
    height = math.ceil(math.log2(len(individual.serialize())))
    penalty = 0 if height < 3 else -sum([math.pow(2, i) for i in range(3, height)])
    return diff + penalty
    # return min(diff, penalty)


# Funcion utilizada como "termination condition"
# Termina cuando el mejor fitness es mayor a -100 (al menos la mitad
# de los x es suficientemente cercano y/o arbol con poco penalty)
def is_done(fitness_values):
    max_fitness = max(fitness_values)
    if max_fitness >= -100:  # suficientemente cercanas
        return fitness_values.index(max_fitness)
    else:
        return None


# Funcion para correr el algoritmo genetico y generar los graficos pedidos.
def run(seq, problem="Symbolic Regression", ps=10, ff=fitness, gg=AST1, tc=is_done, mi=50,
        e=0.2, mr=0.8, md=10, imd=6, fitness_analysis=False, show=True):
    ga = GeneticAlgorithm(population_size=ps, fitness_fun=ff, gene_generator=gg,
                          termination_cond=tc, max_iter=mi, elitism=e, mutation_rate=mr,
                          max_depth=md, initial_max_depth=imd)

    best, iters = ga.run()
    print("seq to find > ", seq)
    print("got: ", best)
    print(f"in {iters} iterations")
    if fitness_analysis:
        bests, avg, worst = ga.get_analysis()
        fitness_per_generation(bests, avg, worst, show=show, problem=problem,
                               to_find=str(seq), got=best)
        fitness_per_generation(best=bests, show=show, problem=problem+' (Mejor)',
                               to_find=str(seq), got=best)
        fitness_per_generation(avg= avg, show=show, problem=problem+' (Promedio)',
                               to_find=str(seq), got=best)
    return best, iters


def run_configurations(seq):
    populations = np.arange(50, 1001, 50)
    mutation_rates = np.arange(0, 1.1, 0.1)
    map_iters = np.empty((np.size(populations), np.size(mutation_rates)))
    map_fitness = np.empty((np.size(populations), np.size(mutation_rates)))
    for i in range(np.size(populations)):
        print(f"- Running for population size = {populations[i]}")
        for j in range(np.size(mutation_rates)):
            print(f"-- Running for mutation rate = {mutation_rates[j]}")
            ga = GeneticAlgorithm(population_size=populations[i], fitness_fun=fitness, gene_generator=AST1,
                                  termination_cond=is_done, max_iter=30, elitism=0, mutation_rate=mutation_rates[j],
                                  max_depth=5, initial_max_depth=2)
            _, iters = ga.run()
            best, _, _ = ga.get_analysis()
            map_iters[i, j] = iters
            map_fitness[i, j] = best[-1]
    heatmap_configurations(populations, mutation_rates, map_iters, type='I',
                           problem="Symbolic Regression", to_find=''.join([str(i) for i in seq]))
    #heatmap_configurations(populations, mutation_rates, map_fitness, type='F',
    #                       problem="Best fitness for Symbolic Regression", to_find=''.join([str(i) for i in seq]))


if __name__ == "__main__":
    # Symbolic regression
    # run("x^2 + x - 6", ff=fitness, fitness_analysis=True, md=2, imd=2, e=0.1, mr=0.7, mi=50, ps=10)
    # Implementando el nodo division
    # run("x^2 + x - 6", ff=fitness, fitness_analysis=True, md=2, imd=2, e=0.1, mr=0.7, mi=50, ps=10,
    #    gg=AST2, problem="Symbolic Regression with Div")

    # Heatmap
    run_configurations("x2+x-6")
