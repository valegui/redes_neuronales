from GeneticAlgorithm import GeneticAlgorithm
from AST import AST
from arboles import *
from utils import fitness_per_generation
import math
import time
import random
import sys
import matplotlib.pyplot as plt

TERMINAL = [25, 7, 8, 100, 4, 2]
AST1 = AST([AddNode, SubNode, MultNode, MaxNode], TERMINAL, 0.5)
AST2 = AST([AddNode, SubNode, MultNode], TERMINAL, 0.5)

GOAL = 65346
NEG_INF = float("-inf")
MIN_SYS = - sys.maxsize - 1


# 2.1.1
# Funcion de fitness para el problema sin limite de repeticiones
def numeric_difference(individual):
    value = individual.eval()
    return - abs(GOAL - value)


# 2.1.2
# Funcion de fitness que permite castigar a los arboles muy grandes
def min_diff_penalty(individual):
    height = math.ceil(math.log2(len(individual.serialize())))
    penalty = 0 if height < 7 else -sum([math.pow(2, i) for i in range(7, height)])
    return numeric_difference(individual) + penalty
#    return min(numeric_difference(individual), penalty)


# 2.1.3
# Funcion de fitness que permite castigar a los arboles muy grandes y a aquellos con terminales repetidos.
def no_repetition(individual):
    nodes = individual.serialize()
    terminals = []
    for node in nodes:
        if type(node) == TerminalNode:
            ne = node.eval()
            if ne not in terminals:
                terminals.append(ne)
            else:
                return MIN_SYS  # NEG_INF
    return min_diff_penalty(individual)


# Funcion utilizada como "termination condition"
# Termina cuando el fitness mas bueno es 0 (el numero es encontrado)
def is_done(fitness_values):
    max_fitness = max(fitness_values)
    if max_fitness >= 0:  # 0 and count_max >= len(fitness_values) * 0.9:
        return fitness_values.index(max_fitness)
    else:
        return None


# Funcion para correr el algoritmo genetico y generar los graficos pedidos.
def run(seq, problem="Encontrar numero", ps=10, ff=numeric_difference, gg=AST1, tc=is_done, mi=50,
        e=0.2, mr=0.8, md=10, imd=6, fitness_analysis=False, show=True):
    ga = GeneticAlgorithm(population_size=ps, fitness_fun=ff, gene_generator=gg,
                          termination_cond=tc, max_iter=mi, elitism=e, mutation_rate=mr,
                          max_depth=md, initial_max_depth=imd)

    best, iters = ga.run()
    print("seq to find > ", seq)
    print("got: ", best.eval())
    print("as: ", best)
    print(f"in {iters} iterations")
    if fitness_analysis:
        bests, avg, worst = ga.get_analysis()
        fitness_per_generation(bests, avg, worst, show=show, problem=problem,
                               to_find=str(seq), got=best.eval())
        fitness_per_generation(avg=avg, show=show, problem=problem+' (Promedio)',
                               to_find=str(seq), got=best.eval())
        fitness_per_generation(best=bests, show=show, problem=problem + ' (Mejor)',
                               to_find=str(seq), got=best.eval())
    return best, iters


if __name__ == "__main__":
    # Sin limite de repeticiones
    random.seed(665)
    run(GOAL, ff=numeric_difference, gg=AST1, fitness_analysis=True, md=10, imd=2, e=0.2, mr=0.7, mi=50, ps=14,
        problem="Encontrar numero - con repeticion")

    # Fitness
    random.seed(5)
    run(GOAL, ff=min_diff_penalty, gg=AST1, fitness_analysis=True, md=10, imd=2, e=0.2, mr=0.7, mi=50, ps=14,
        problem="Encontrar numero - penalizar alto")

    # Sin repeticion
    random.seed(6)
    run(GOAL, ff=no_repetition, gg=AST2, fitness_analysis=True, md=6, imd=2, e=0.2, mr=0.7, mi=50, ps=14,
        problem="Encontrar numero - sin repeticion")

    # Comparacion de tiempo entre 2.1.1 vs 2.1.2
    times_1 = []
    best_fitness_1 = []
    times_2 = []
    best_fitness_2 = []
    for _ in range(15):
        seed = random.random()

        random.seed(3 * seed)
        start_1 = time.time()
        best_1, _ = run(GOAL, ff=numeric_difference, fitness_analysis=False, md=15, imd=6, e=0.2, mr=0.7, mi=50, ps=14)
        end_1 = time.time()
        times_1.append(end_1-start_1)
        best_fitness_1.append(best_1.eval())

        random.seed(3 * seed)
        start_2 = time.time()
        best_2, _ = run(GOAL, ff=min_diff_penalty, fitness_analysis=False, md=15, imd=6, e=0.2, mr=0.7, mi=50, ps=14)
        end_2 = time.time()
        times_2.append(end_2 - start_2)
        best_fitness_2.append(best_2.eval())

    plt.subplot(2, 1, 1)
    plt.plot(times_1, 'b--', label='Fitness sin penalty')
    plt.plot(times_2, 'r--', label='Fitness con penalty')
    plt.title('Tiempo y valor obtenidos con fitness que (no) \npenalizan por la altura del arbol')
    plt.ylabel('Tiempo [s]')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(best_fitness_1, 'b--', label='Fitness sin penalty')
    plt.plot(best_fitness_2, 'r--', label='Fitness con penalty')
    plt.plot([GOAL] * 15, 'g-.', label='GOAL')
    plt.ylabel('Valor')
    plt.legend()
    plt.savefig(f"analysis/time_comp_encontrarnumero_{GOAL}.png")
    plt.show()





