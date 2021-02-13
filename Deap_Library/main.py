import pandas as pd
import random
import time
from statistics import mean, stdev

import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools


def individual(icls):
    genome = list()
    genome.append(random.uniform(-10, 10))
    genome.append(random.uniform(-10, 10))

    return icls(genome)


def fitness_function(individual):
    result = (individual[0] + 2 * individual[1] - 7) ** 2 + (2 * individual[0] + individual[1] - 5) ** 2
    return result,


def toPlot(highest_res, mean_res, std):
    fig = plt.figure(1)
    plt.plot(highest_res)

    plt.title("Wartość funkcji od iteracji", fontsize=16)
    plt.ylabel("wartości", fontsize=14)
    plt.xlabel("epoki", fontsize=14)
    fig.savefig('val.png')

    fig2 = plt.figure(2)
    plt.plot(mean_res)

    plt.title("Średnie wartości funkcji", fontsize=16)
    plt.ylabel("wartości", fontsize=14)
    plt.xlabel("epoki", fontsize=14)
    fig2.savefig('mean.png')

    fig3 = plt.figure(3)
    plt.plot(std)

    plt.title("Odchylenie standardowe od iteracji", fontsize=16)
    plt.ylabel("wartości", fontsize=14)
    plt.xlabel("epoki", fontsize=14)
    fig3.savefig('std.png')


def genetic_algorithm():
    tstart = time.time()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('individual', individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_function)
    toolbox.register("select", tools.selTournament, tournsize = 3)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu = 5, sigma = 10, indpb = 0.4)

    sizePopulation = 200
    probabilityMutation = 0.2
    probabilityCrossover = 0.8
    numberIteration = 200

    pop = toolbox.population(n=sizePopulation)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.value = fit

    g = 0
    numberElitism = 1
    bestlist = []
    meanlist = []
    stdlist = []
    while g < numberIteration:
        g = g + 1
        print("--- Epoch %i ---" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < probabilityCrossover:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < probabilityMutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print(" Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean1 = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean1 ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean1)
        print("  Std %s" % std)
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        bestlist.append(min(fits))
        meanlist.append(mean(bestlist))
        if g > 1:
            stdlist.append(stdev(bestlist))

    tend = time.time()
    t = tend - tstart
    #toPlot(bestlist,meanlist,stdlist)
    return min(bestlist), t


if __name__ == '__main__':
    best = []
    timeres = []
    for i in range(1):
        res, t = genetic_algorithm()
        best.append(res)
        timeres.append(t)
    print("Wynik: "+ str(sum(best) / len(best)))
    print("Czas: " + str(sum(timeres) / len(timeres)))

print("-- End of succesful evolution --")
