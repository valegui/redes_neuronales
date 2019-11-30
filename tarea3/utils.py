import matplotlib.pyplot as plt
import seaborn as sns


def fitness_per_generation(best=None, avg=None, worst=None, show=True, problem=None, to_find=None, got=None):
    if best is not None:
        plt.plot(best, '-g', label="Mejor fitness")
    if avg is not None:
        plt.plot(avg, '-b', label="Fitness promedio")
    if worst is not None:
        plt.plot(worst, '-r', label="Peor fitness")
    plt.legend()
    title = "Fitness por generacion"
    if problem is not None:
        title += f"\n {problem} : {to_find}"
    if got is not None:
        title += f"\n got : {got}"
    plt.title(title)
    plt.ylabel("Fitness")
    plt.xlabel("Generacion")
    plt.savefig(f"analysis/fit_gen_{problem.replace(' ','')}_{to_find}.png")
    if show:
        plt.show()


def heatmap_configurations(populations, mutation_rates, heatmap_data, problem=None, to_find=None, type=None):
    sns.palplot(sns.cubehelix_palette(8))
    cbar_label = 'Number of Iterations' if type == 'I' else 'Best Fitness'
    fig = plt.figure(figsize=(8.3, 8.3))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    sns.heatmap(heatmap_data, ax=ax, cbar_kws={'label': cbar_label})
    ax.set(yticklabels=[int(number) for number in populations])
    ax.set(xticklabels=["{:.1f}".format(number) for number in mutation_rates])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    type_name = 'number of iterations to get the result' if type == 'I' else 'best fitness when terminated'
    if problem is not None:
        plt.title(f"Heatmap of {type_name} \n {problem} : {to_find}", fontsize=12)
    else:
        plt.title(f"Heatmap of {type_name}")
    plt.xlabel("Mutation rate", fontsize=12)
    plt.ylabel("Population size", fontsize=12)
    plt.savefig(f"analysis/heatmap_configurations_{type}_{to_find}.png")
    plt.show()
