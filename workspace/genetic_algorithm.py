import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loaders import *
import yaml
from yaml import load
from enum import Enum

class Component(Enum):
    DRAM: 1 # maybe 0 icr
    BUFFER: 3
    PE: 4

# pe_dims = {'pe_meshX': 1, 'pe_meshY': 16}
# pe_dims={'pe_meshX': 2, 'pe_meshY': 8}
# pe_dims = {'pe_meshX': 4, 'pe_meshY': 4}

class GeneticAlgorithm:

    def __init__(
            self, 
            dataflow=['R', 'S', 'P', 'Q', 'C', 'M', 'N'], 
            components=[4], #Component.PE,
            workload='layer_shapes/fc1.yaml', 
            pe_dims={'pe_meshX': 2, 'pe_meshY': 8}, 
            mapper='designs/_include/mapper.yaml',
            constraints='designs/system/constraints.yaml',
            architecture='designs/system/arch.yaml',
            n=5, 
            k=20, 
            p=10, 
            iter=10,
            early_stop=2,
            known_values={}
            ):
        # convolution
        self.dataflow = dataflow
        self.components=components
        self.workload = workload
        self.pe_dims = pe_dims

        self.mapper=mapper
        self.constraints=constraints
        self.architecture=architecture
        
        # n population -> k mutations -> p selection -> n population
        # constraints: n | k, p/2 = n
        self.n = n
        self.k = k
        self.p = p
        self.iter = iter
        self.early_stop=early_stop
        
        self.mapper_call_count = 0
        self.VISITED={}
        self.known_values=known_values
        self.mapper_calls = [] # list of [permutation, fitness] in order of call
        self.selected_fitnesses = [] # list of [permutation, fitness] for all selected in each iter in order

    def dummy_fitness(self, dataflow):
        return random.uniform(0, 1.0e-7)  
        
    
    def fitness(self, dataflow):
        '''
        Calculates the fitness of a given permustation `dataflow`. Returns the
        inverse EDP after running the timeloop mapper on this dataflow.
        '''
        
        permutation = ''.join(dataflow)
        
        if permutation in self.known_values: # im trying to save time
            print(f'{permutation} in known values!')
            inverse_EDP = self.known_values[permutation]
        else: # call mapper
            data = self.evaluate(dataflow)
            energy, latency = data
            inverse_EDP = 1 / (energy * latency)
            self.known_values[permutation] = inverse_EDP # update our yaml file

        # keeping track of mapper call count for data purposes
        self.mapper_call_count += 1
        
        # update visited
        self.VISITED[permutation] = inverse_EDP # update this instance's visited
        self.mapper_calls.append([permutation, inverse_EDP])
            
        print(f"{dataflow} has a fitness of {inverse_EDP}")
        return inverse_EDP


    def evaluate(self, dataflow):
        '''
        Evaluates the given dataflow on -- architecture

        dataflow: computation ordering in list format
        workload: the file path to the workload this is being evaluated on
        returns tuple of energy, latency
        '''
        filename = ''.join(dataflow)

        stream = open(self.constraints, 'r')
        dictionary = yaml.safe_load(stream)

        for component in self.components:
            idx = component # 4 is PE
            dictionary['constraints']['targets'][idx]['permutation'] = dataflow

        with open(f'iters/configs/{filename}.yaml', 'w') as file:
            yaml.dump(dictionary, file, default_flow_style=False)

        constraints = f'iters/configs/{filename}.yaml'

        result = run_timeloop_mapper(
            # config,
            self.pe_dims,
            architecture=self.architecture,
            mapper=self.mapper,
            problem=self.workload,
            constraints=constraints 
        )
        
        stats = open('./output_dir/timeloop-mapper.stats.txt', 'r').read()
        mapping = result.mapping

        lines = stats.split('\n')
        energy = float([l for l in lines if 'Energy:' in l][0].split(' ', 2)[1])
        cycles = int([l for l in lines if 'Cycles:' in l][0].split(' ', 1)[1])

        print(energy, cycles)
        return energy, cycles


    def mutation(self, population):
        """
        Performs a random swap mutation for every member of `population`.
        Returns a list of the new mutated population.
        """
        mutations = []

        for df in population:
            for _ in range(self.k // self.n):
                mutation = df.copy()
                # swap two random indices
                idx1, idx2 = random.sample(range(len(self.dataflow)), 2)
                mutation[idx1], mutation[idx2] = mutation[idx2], mutation[idx1]
                mutations.append(mutation)
        return mutations


    def selection(self, population: list[str], p: int, fitness, workload: str, pe_dims) -> list:
        """
        Evaluates the fitness using `fitness` func of all members of a population on
        `workload` workload with pe dimensions `pe_dims`, and performs selection of 
        the top p candidates.
        """
        print("selection")
        fitnesses = []
        i = 0
        for candidate in population:
            i += 1
            print(f'{i}/{len(population)} candidate')
            # do a check to see if we've already run this config
            trial_name = ''.join(candidate)
            if trial_name in self.VISITED:
                print('already visited ' + trial_name)
                fitnesses.append([candidate, self.VISITED[trial_name]])
            else:
                fitnesses.append([candidate, fitness(candidate)])

        fitnesses.sort(key=lambda x: x[1], reverse=True) # high to low fitness
        selections = fitnesses[:p] 
        selections = [x[0] for x in selections] # len(selections) = p
        return selections

    
    def crossover(self, population) -> list:
        print("crossover")
        random.shuffle(population)
        crossover_pairs = [(population[i], population[i+1]) for i in range(0, len(population), 2)]
        crossovers = []  # len(crossovers) = n
        for pair in crossover_pairs:
            s1, s2 = pair
            cut_point = random.randint(1, len(s1) - 1) # split at a random point and join
            first_half = s1[:cut_point]
            second_half = s2.copy()

            for parameter in first_half:
                second_half.remove(parameter)
            crossover = first_half + second_half
            crossovers.append(crossover)

        return crossovers


    def run(self, g=6.0e-8):
        
        # fitness = self.dummy_fitness # TODO: FOR DEBUGGING
        fitness = self.fitness

        # Generate n base permutations
        population = [random.sample(self.dataflow, len(self.dataflow)) for _ in range(self.n)]

        print("Initializing")
        # Initialize base fitness and goal fitness
        dfs_fitnesses = [[df, fitness(df)] for df in population]
        best_df, f = max(dfs_fitnesses, key=lambda x: x[1])

        plateau = 0
        for i in range(self.iter):
            print("\nITERATION: ", i)
            plateau += 1
            
            # Mutation
            mutations = self.mutation(population)
            # print(mutations)
            
            # Selection
            selections = self.selection(mutations, self.p, fitness, self.workload, self.pe_dims)
            # print(f'selections done running: {selections}')

            # TODO: we should also evaluate the best fitness here before crossing over or just add the selections to crossovers

            # Crossover
            crossovers = self.crossover(selections)
            # print(f'crossovers done running: {crossovers}')

            crossovers.extend(selections)
            crossovers_fitnesses = []
            i = 0
            for crossover in crossovers:
                i += 1
                print(f'{i}/{len(crossovers)} candidate in crossover')
                trial_name = ''.join(crossover)
                if trial_name in self.VISITED:
                    print(f'already visited {trial_name}')
                    crossovers_fitnesses.append([crossover, self.VISITED[trial_name]])
                else: 
                    crossovers_fitnesses.append([crossover, fitness(crossover)])
            # print(crossovers_fitnesses)

            # UPDATE POPULATION FOR NEXT ROUND!!!!!!
            crossovers_fitnesses.sort(key=lambda x: x[1], reverse=True) # high to low fitness
            top_n_fitnesses = crossovers_fitnesses[:self.n] 
            # print(top_n_fitnesses)
            self.selected_fitnesses.extend([x[1] for x in top_n_fitnesses])
            population = [x[0] for x in top_n_fitnesses]
            # print(population)

            best_df_trial, f_trial = max(crossovers_fitnesses, key=lambda x: x[1])
            if f_trial > f:
                plateau = 0
                best_df, f = best_df_trial, f_trial
                print(f'new best trial: {best_df_trial} with fitness {f}')
            if f >= g:
                # goal fitness reached, break
                print('reached goal fitness, returning')
                break
            if plateau >= self.early_stop:
                print(f'early stop after plateauing for {plateau} iters')
                break

        return best_df, f



    def run_separate():
        fitness = self.fitness

        populations = {}
        for component in self.components:
            # generate a set of random base permutations for each component
            populations[component] = [random.sample(self.dataflow, len(self.dataflow)) for _ in range(self.n)]

        print("Initializing")
        # Initialize base fitness and goal fitness
        dfs_fitnesses = [[df, fitness(df)] for df in population] # TODO: need to update fitness to take in all permutatinos
        best_df, f = max(dfs_fitnesses, key=lambda x: x[1])

        plateau = 0

        for i in range(self.iter):
            print("\nITERATION: ", i)
            plateau += 1
            
            # Mutation
            for component in self.components:
                mutations = self.mutation(populations[component])
            # print(mutations)
            
            # Selection
            selections = self.selection(mutations, self.p, fitness, self.workload, self.pe_dims) # TODO: need to update selection to take all mutations
            # also would like selections to return a dict here like populations

            # Crossover
            crossovers = {}
            for component in self.components:
                crossovers[component] = self.crossover(selections[component])

            crossovers.extend(selections) # TODO fix this
            crossovers_fitnesses = []
            i = 0
            for crossover in crossovers:
                i += 1
                print(f'{i}/{len(crossovers)} candidate in crossover')
                trial_name = ''.join(crossover)
                if trial_name in self.VISITED:
                    print(f'already visited {trial_name}')
                    crossovers_fitnesses.append([crossover, self.VISITED[trial_name]])
                else: 
                    crossovers_fitnesses.append([crossover, fitness(crossover)])
            # print(crossovers_fitnesses)

            # UPDATE POPULATION FOR NEXT ROUND!!!!!!
            crossovers_fitnesses.sort(key=lambda x: x[1], reverse=True) # high to low fitness
            top_n_fitnesses = crossovers_fitnesses[:self.n] 
            # print(top_n_fitnesses)
            self.selected_fitnesses.extend([x[1] for x in top_n_fitnesses])
            population = [x[0] for x in top_n_fitnesses]
            # print(population)

            best_df_trial, f_trial = max(crossovers_fitnesses, key=lambda x: x[1])
            if f_trial > f:
                plateau = 0
                best_df, f = best_df_trial, f_trial
                print(f'new best trial: {best_df_trial} with fitness {f}')
            if f >= g:
                # goal fitness reached, break
                print('reached goal fitness, returning')
                break
            if plateau >= self.early_stop:
                print(f'early stop after plateauing for {plateau} iters')
                break

        return best_df, f
        
