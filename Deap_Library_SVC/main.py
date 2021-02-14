import pandas as pd
import random
import math
import time
from statistics import mean, stdev

import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def SVCParameters(numberFeatures, icls):
    genome = list()

    # kernel
    listKernel = ["linear", "rbf", "poly", "sigmoid"]
    genome.append(listKernel[random.randint(0, 3)])

    # c
    k = random.uniform(0.1, 100)
    genome.append(k)

    # degree
    genome.append(random.uniform(0.1, 5))

    # gamma
    gamma = random.uniform(0.001, 5)
    genome.append(gamma)

    # coeff = random.uniform(0.01, 10)
    # genome.append(coeff)

    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))

    return icls(genome)


def SVCParametersFitness(y, df, numberOfAttributes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    list_columns_to_drop = []
    for i in range(numberOfAttributes, len(individual)):
        if individual[i] == 0:
            list_columns_to_drop.append(i - numberOfAttributes)

    df_selected_features = df.drop(df.columns[list_columns_to_drop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df_selected_features)

    estimator = SVC(kernel=individual[0], C=individual[1], degree=individual[2], gamma=individual[3],
                    coef0=individual[4], random_state=101)
    result_sum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        result_sum = result_sum + result

    return result_sum / split,


def mutationSVC(individual):
    number_paramer = random.randint(0, len(individual) - 1)
    if number_paramer == 0:
        # kernel
        listKernel = ["linear", "rbf", "poly", "sigmoid"]
        individual[0] = listKernel[random.randint(0, 3)]
    elif number_paramer == 1:
        # c
        k = random.uniform(0.1, 100)
        individual[1] = k
    elif number_paramer == 2:
        # degree
        individual[2] = random.uniform(0.1, 5)
    elif number_paramer == 3:
        # gamma
        gamma = random.uniform(0.01, 1)
        individual[3] = gamma
    elif number_paramer == 4:
        coeff = random.uniform(0.01, 1)
        individual[4] = coeff
    else:
        if individual[number_paramer] == 0:
            individual[number_paramer] = 1
        else:
            individual[number_paramer] = 0


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


def genetic_algorithm(y, df, numberOfAttributes):
    tstart = time.time()
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('individual', SVCParameters, numberOfAttributes, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", SVCParametersFitness, y, df, numberOfAttributes)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutationSVC)

    sizePopulation = 100
    probabilityMutation = 0.15
    probabilityCrossover = 0.85
    numberIteration = 100

    pop = toolbox.population(n=sizePopulation)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

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
        best_ind = tools.selWorst(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        bestlist.append(max(fits))
        meanlist.append(mean(bestlist))
        if g > 1:
            stdlist.append(stdev(bestlist))

    tend = time.time()
    t = tend - tstart
    toPlot(bestlist, meanlist, stdlist)
    return bestlist.index(max(bestlist)), max(bestlist), t


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("bcancer.csv", sep=',')

    y = df['Classification']
    df.drop('Classification', axis=1, inplace=True)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    # clf = SVC()
    # scores = model_selection.cross_val_score(clf, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)

    numberOfAttributes = len(df.columns)
    ind, res, t = genetic_algorithm(y, df, numberOfAttributes)

    print("Best result --- " + str(res))
    print("Epoch --- " + str(ind + 1))
    print("Overall Time --- " + str(t))
