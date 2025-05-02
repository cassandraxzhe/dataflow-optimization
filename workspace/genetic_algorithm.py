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
            component=Component.PE,
            workload='layer_shapes/fc1.yaml', 
            pe_dims={'pe_meshX': 2, 'pe_meshY': 8}, 
            mapper='designs/_include/mapper.yaml',
            constraints='designs/system/constraints.yaml',
            architecture='designs/system/arch.yaml',
            n=5, 
            k=20, 
            p=10, 
            iter=10,
            early_stop=2
            ):
        # convolution
        self.dataflow = dataflow
        self.component=component
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
        self.mapper_calls = [] # list of [permutation, fitness] in order of call
        self.selected_fitnesses = [] # list of [permutation, fitness] for all selected in each iter in order

    def dummy_fitness(dataflow):
        return random.uniform(0, 1.0e-7)  
        
    
    def fitness(self, dataflow):
        '''
        Calculates the fitness of a given permustation `dataflow`. Returns the
        inverse EDP after running the timeloop mapper on this dataflow.
        '''
        data = self.evaluate(dataflow)
        energy, latency = data
        inverse_EDP = 1 / (energy * latency)
        print(f"{dataflow} has a fitness of {inverse_EDP}")

        # update visited
        self.VISITED[''.join(dataflow)] = inverse_EDP
        self.mapper_calls.append([''.join(dataflow), inverse_EDP])
        return inverse_EDP


    def evaluate(self, dataflow):
        '''
        Evaluates the given dataflow on -- architecture

        dataflow: computation ordering in list format
        workload: the file path to the workload this is being evaluated on
        returns tuple of energy, latency
        '''

        stream = open(self.constraints, 'r')
        dictionary = yaml.safe_load(stream)
        idx = self.component # 4 # PE
        dictionary['constraints']['targets'][idx]['permutation'] = dataflow

        filename = ''.join(dataflow)
        with open(f'iters/configs/{filename}.yaml', 'w') as file:
            yaml.dump(dictionary, file, default_flow_style=False)

        constraints = f'iters/configs/{filename}.yaml'

        sys_1x16_result = run_timeloop_mapper( # TODO: this should be run_timeloop_mapper not run_timeloop_model!
            # config,
            self.pe_dims,
            architecture=self.architecture,
            mapper=self.mapper,
            problem=self.workload,
            constraints=constraints 
        )

        # keeping track of mapper call count for data purposes
        self.mapper_call_count += 1
        
        stats = open('./output_dir/timeloop-mapper.stats.txt', 'r').read()
        mapping = sys_1x16_result.mapping

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
        # print(f'fitnesses: {fitnesses}')
        fitnesses.sort(key=lambda x: x[1], reverse=True) # high to low fitness
        # print(f'sorted fitnesses: {fitnesses}')
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
            # second_half.remove(parameter for parameter in first_half)
            for parameter in first_half:
                second_half.remove(parameter)
            crossover = first_half + second_half
            crossovers.append(crossover)

        return crossovers


    # TODO: include condition to check if a permutation has already been tested
    # TODO: why aren't we reaching crossover print statement?
    # TODO: why isn't the best fitness from initial population being chosen? ??????????


    def run(self, g=6.0e-8):
        
        # fitness = dummy_fitness # TODO: FOR DEBUGGING
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

            fitnesses = crossovers_fitnesses.sort(key=lambda x: x[1], reverse=True) # high to low fitness
            # print(f'sorted fitnesses: {fitnesses}')
            top_n_fitnesses = fitnesses[:n] 
            population = [x[0] for x in selections] # len(selections) = p

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
        
