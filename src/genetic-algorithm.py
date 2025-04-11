import random
import numpy as np



params = ['M', 'N', 'K', 'C', 'P', 'Q']


def evaluate_fitness(child: str) -> int:
    return 0;


def crossover(parent1: str, parent2: str) -> str:
    # Randomly select a crossover point
    crossover_point = random.randint(1, len(parent1) - 1)
    # Create the child by combining parts of both parents
    child = parent1[:crossover_point] + parent2[crossover_point:]
    # TODO: make sure it's a valid child
    return child


def mutate(child: str) -> str:
    # Randomly select a mutation point
    mutation_point = random.randint(0, len(child) - 1)
    # Randomly select a new character
    new_char = random.choice(params)
    # Create the mutated child
    mutated_child = child[:mutation_point] + new_char + child[mutation_point + 1:]
    return mutated_child



def genetic_algorithm(goalFitness: int = 0, params: list = params, iterations: int = 100) -> str:

    # Generate a list of random starting strings from the given loops
    to_use = params.copy()
    parents = []
    for i in range(10):
        # Generate a random string of length 6
        random.shuffle(to_use)
        string = "".join(to_use)
        parents.append(string)


    goalFitnessAcheived = False
    i = 0
    most_fit = None

    # for each generation
    while (not goalFitnessAcheived | i < iterations):
        # Generate the 20 children
        children = []
        fitnesses = {}

        # evaluate the fitness of each child
        for child in children:
            fitness = evaluate_fitness(child)
            fitnesses[child] = fitness

        # sort the children by fitness
        sorted_children = sorted(children, key=lambda x: fitnesses[x], reverse=True)
        # select the top 10 children
        selected_children = sorted_children[:10]

        # check if the goal fitness has been reached 
        if fitnesses[selected_children[0]] >= goalFitness:
            goalFitnessAcheived = True
            most_fit = selected_children[0]
            break

        # create the next generation
        next_generation = []
        for i in range(0, len(selected_children), 2):
            parent1 = selected_children[i]
            parent2 = selected_children[i + 1]
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)

        # replace the old generation with the new generation
        parents = next_generation
        i += 1

    return most_fit


