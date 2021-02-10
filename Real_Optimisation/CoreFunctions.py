import random
import math
import sys
import numpy as np


def generate_populations(population_length):
    pops = [[], []]
    pops[0] = [random.uniform(-10, 10) for i in range(population_length)]
    pops[1] = [random.uniform(-10, 10) for i in range(population_length)]
    return pops


def function(populations):
    res = []
    for i in range(len(populations[0])):
        x1 = populations[0][i]
        x2 = populations[1][i]
        y = ((x1 + (2 * x2) - 7) ** 2) + (((2 * x1) + x2 - 5) ** 2)
        res.append(y)
    return res


def sort(function_values, populations, maximum):
    newPopulations = [[], []]

    newFunctionValues, newPopulations[0], newPopulations[1] = (list(t) for t in zip(
        *sorted(zip(function_values, populations[0], populations[1]), reverse=maximum)))
    return newFunctionValues, newPopulations


def select(populations, function_values, minmax, selection_type, selection_probability):
    min = False
    if minmax == "maximisation":
        min = True

    if selection_type == "best":
        res_populations = [[], []]
        new_length = math.ceil(len(populations[0]) * float(selection_probability))
        new_function, new_populations = sort(function_values, populations, min)
        for i in range(new_length):
            res_populations[0].append(new_populations[0][i])
            res_populations[1].append(new_populations[1][i])
        return res_populations

    elif selection_type == "roulette":
        if not min:
            function_values = [1 / f for f in function_values]

        function_sum = float(sum(function_values))
        values_prob = [value / function_sum for value in function_values]
        probs = [sum(values_prob[:i + 1]) for i in range(len(values_prob))]
        res_populations = [[], []]

        for n in range(int(len(function_values))):
            r = np.random.rand(1)
            for i in range(len(function_values)):
                if r <= probs[i]:
                    res_populations[0].append(populations[0][i])
                    res_populations[1].append(populations[1][i])
                    break
        return res_populations

    else:
        if min:
            function_values = [1 / f for f in function_values]
        indexlist = []
        for i in range(len(function_values)):
            indexlist.append(i)
        winners = []
        groups = []
        res_populations = [[], []]
        for i in range(int(selection_probability)):
            group = []
            for j in range(int(len(populations[0]) / int(selection_probability))):
                rand_numb = random.randrange(0, len(indexlist))
                index = indexlist[rand_numb]
                indexlist.pop(rand_numb)
                group.append(index)
            groups.append(group)
        for i in groups:
            winners.append(fight(i, function_values))
        for i in range(len(winners)):
            res_populations[0].append(populations[0][winners[i]])
            res_populations[1].append(populations[1][winners[i]])
        return res_populations


def fight(group, values):
    best = sys.maxsize
    best_id = 0
    for i in group:
        if values[i] < best:
            best = values[i]
            best_id = i
    return best_id


def crossover(pops, populations, probability, type):
    res_populations = [[], []]

    while True:
        r1 = random.randint(0, len(populations))
        parent1_1 = populations[0][r1]
        parent1_2 = populations[1][r1]
        r2 = random.randint(0, len(populations))
        parent2_1 = populations[0][r2]
        parent2_2 = populations[1][r2]

        prob = np.random.rand(1)
        if prob <= float(probability):
            if type == "arithmetic":
                child1_1, child1_2, child2_1, child2_2 = cross_parents_arithmetic(parent1_1, parent1_2, parent2_1,
                                                                                  parent2_2)
                res_populations[0].append(child1_1)
                res_populations[1].append(child1_2)
                res_populations[0].append(child2_1)
                res_populations[1].append(child2_2)
            else:
                child1_1, child2_1 = cross_parents_heuristic(parent1_1, parent1_2, parent2_1,
                                                             parent2_2)
                if child1_1 != sys.maxsize:
                    res_populations[0].append(child1_1)
                    res_populations[1].append(child2_1)
        else:
            res_populations[0].append(parent1_1)
            res_populations[1].append(parent2_1)

        if len(res_populations[0]) >= len(pops[0]):
            break
    return res_populations


def cross_parents_arithmetic(x1, y1, x2, y2):
    k = np.random.rand(1)
    x1_new = (k * x1) + ((1 - k) * x2)
    y1_new = (k * y1) + ((1 - k) * y2)
    x2_new = ((1 - k) * x1) + (k * x2)
    y2_new = ((1 - k) * y1) + (k * y2)
    return x1_new[0], y1_new[0], x2_new[0], y2_new[0]


def cross_parents_heuristic(x1, y1, x2, y2):
    if x2 >= x1 and y2 >= y1:
        k = np.random.rand(1)
        x1_new = k * (x2 - x1) + x1
        y1_new = k * (y2 - y1) + y1
        return x1_new[0], y1_new[0]
    else:
        return sys.maxsize, 0


def mutation(populations, probability):
    res_populations = populations
    for i in range(len(res_populations)):
        prob = np.random.rand(1)
        index = random.randint(0, 1)
        if prob <= float(probability):
            res_populations[index][i] = random.uniform(-10, 10)
    return res_populations
