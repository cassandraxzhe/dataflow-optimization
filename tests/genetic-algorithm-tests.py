import unittest
from . import DONT_USE_genetic_algorithm as ga
import random

class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.params = ['A', 'B', 'C', 'D', 'E', 'F']
        self.goal_fitness = 6
        self.iterations = 100
        self.ga = ga.genetic_algorithm

    # def test_evaluate_fitness(self):
    #     # Test with a string that matches the goal fitness
    #     test_string = "ABCDEF"
    #     fitness = ga.evaluate_fitness(test_string)
    #     self.assertEqual(fitness, 6)

    #     # Test with a string that does not match the goal fitness
    #     test_string = "ABCXYZ"
    #     fitness = ga.evaluate_fitness(test_string)
    #     self.assertEqual(fitness, 3)

    def test_crossover_1(self):
        parent1 = "ABCDEF"
        parent2 = "EFACBD"
        child = ga.crossover(parent1, parent2)
        self.assertEqual(len(child), len(parent1))
        for char in parent1:
            self.assertIn(char, child, f"{char} not in child! Invalid child: {child}")

    
    def test_crossover_2(self):
        parent1 = "MNKPCQ"
        parent2 = "MNKPCQ"
        child = ga.crossover(parent1, parent2)
        self.assertEqual(len(child), len(parent1))
        self.assertEqual(child, parent1, f"Child should be identical to parent1: {parent1} != {child}")

    def test_mutate(self):
        child = "ABCDEF"
        mutated_child = ga.mutate(child)
        self.assertNotEqual(child, mutated_child)
        self.assertEqual(len(child), len(mutated_child), f"Mutated child should have same length as original child: {child} != {mutated_child}")
        for char in child:
            self.assertIn(char, mutated_child, f"{char} not in mutated child! Invalid child: {mutated_child}")
