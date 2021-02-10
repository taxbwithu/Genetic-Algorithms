import random
import math
import sys
import numpy as np


def calc_sections(acc):
    return math.ceil(math.log2(20 * (10 ** acc)) + math.log2(1))


def generate_populations(chain_sections, population_length):
    pops = [[], []]
    pops[0] = [[random.randint(0, 1) for x in range(chain_sections)] for i in range(population_length)]
    pops[1] = [[random.randint(0, 1) for x in range(chain_sections)] for i in range(population_length)]
    return pops


def bin_to_dec(pop, chain_sections):
    temp_string = ""
    x = temp_string.join(map(str, pop))
    return (-10) + int(x, 2) * 20 / (2 ** chain_sections - 1)


def function(populations, chain_sections):
    res = []
    for i in range(len(populations[0])):
        x1 = bin_to_dec(populations[0][i], chain_sections)
        x2 = bin_to_dec(populations[1][i], chain_sections)
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
        if (values[i] < best):
            best = values[i]
            best_id = i
    return best_id


def crossover(pops, populations, elite, probability, type):
    while True:
        r1 = random.randint(0, len(populations))
        parent1_1 = populations[0][r1]
        parent1_2 = populations[1][r1]
        r2 = random.randint(0, len(populations))
        parent2_1 = populations[0][r2]
        parent2_2 = populations[1][r2]

        prob = np.random.rand(1)
        if prob <= float(probability):
            child1_1, child1_2 = cross_parents(type, parent1_1, parent1_2)
            child2_1, child2_2 = cross_parents(type, parent2_1, parent2_2)
        else:
            child1_1 = parent1_1
            child1_2 = parent1_2
            child2_1 = parent2_1
            child2_2 = parent2_2

        elite[0].append(child1_1)
        elite[1].append(child1_2)
        elite[0].append(child2_1)
        elite[1].append(child2_2)
        if len(elite[0]) >= len(pops[0]):
            break
    return elite


def cross_parents(type, parent_1, parent_2):
    child_1 = []
    child_2 = []
    if (type == "one_point"):
        crosspoint = random.randint(1, len(parent_1) - 1)
        for i in range(0, crosspoint):
            child_1.append(parent_1[i])
            child_2.append(parent_2[i])
        for i in range(crosspoint, len(parent_1)):
            child_1.append(parent_2[i])
            child_2.append(parent_1[i])
        return child_1, child_2
    elif (type == "two_point"):
        crosspoint1 = random.randint(1, len(parent_1) - 1)
        crosspoint2 = random.randint(crosspoint1, len(parent_1) - 1)
        for i in range(0, crosspoint1):
            child_1.append(parent_1[i])
            child_2.append(parent_2[i])
        for i in range(crosspoint1, crosspoint2):
            child_1.append(parent_2[i])
            child_2.append(parent_1[i])
        for i in range(crosspoint2, len(parent_1)):
            child_1.append(parent_1[i])
            child_2.append(parent_2[i])
        return child_1, child_2
    else:
        crosspoint1 = random.randint(1, len(parent_1) - 1)
        crosspoint2 = random.randint(crosspoint1, len(parent_1) - 1)
        crosspoint3 = random.randint(crosspoint2, len(parent_1) - 1)
        for i in range(0, crosspoint1):
            child_1.append(parent_1[i])
            child_2.append(parent_2[i])
        for i in range(crosspoint1, crosspoint2):
            child_1.append(parent_2[i])
            child_2.append(parent_1[i])
        for i in range(crosspoint2, crosspoint3):
            child_1.append(parent_1[i])
            child_2.append(parent_2[i])
        for i in range(crosspoint3, len(parent_1)):
            child_1.append(parent_2[i])
            child_2.append(parent_1[i])
        return child_1, child_2


def mutation(populations, probability, type):
    res_populations = [[], []]
    for i in range(len(populations[0])):
        prob = np.random.rand(1)
        if prob <= float(probability):
            child1 = mutate(type, populations[0][i])
        else:
            child1 = populations[0][i]
        prob = np.random.rand(1)
        if prob <= float(probability):
            child2 = mutate(type, populations[1][i])
        else:
            child2 = populations[1][i]
        res_populations[0].append(child1)
        res_populations[1].append(child2)
    return res_populations


def mutate(type, child):
    res_child = child
    if (type == "marginal"):
        endpoint = len(child) - 1
        res_child[endpoint] = 1 if child[endpoint] == 0 else 1
    elif (type == "one_point"):
        onepoint = random.randint(0, len(child) - 1)
        res_child[onepoint] = 1 if child[onepoint] == 0 else 1
    else:
        onepoint = random.randint(0, len(child) - 1)
        twopoint = random.randint(0, len(child) - 1)
        res_child[onepoint] = 1 if child[onepoint] == 0 else 1
        res_child[twopoint] = 1 if child[twopoint] == 0 else 1
    return res_child


def inversion(populations, probability):
    res_populations = [[], []]
    for i in range(len(populations[0])):
        prob = np.random.rand(1)
        if prob <= float(probability):
            child1 = invert(populations[0][i])
        else:
            child1 = populations[0][i]
        prob = np.random.rand(1)
        if prob <= float(probability):
            child2 = invert(populations[1][i])
        else:
            child2 = populations[1][i]
        res_populations[0].append(child1)
        res_populations[1].append(child2)
    return res_populations


def invert(child):
    res_child = child
    onepoint = random.randint(0, len(child) - 1)
    twopoint = random.randint(onepoint, len(child) - 1)
    for i in range(onepoint, twopoint):
        res_child[i] = 1 if child[i] == 0 else 1
    return res_child
