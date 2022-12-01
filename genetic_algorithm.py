from typing import List, Tuple
import random
import numpy as np
from Chord import Chord


class GeneticAlgorithm:
    """
    Individual - list of chords, each chord is a list of three notes. Length of this chord are one quarter
    Population - list of individuals
    Crossover type - two-point crossover
    Mutation - Random Resetting Mutation
    Selection - Roulette Wheel Selection
    """
    CHORD_SIZE = 3

    def __init__(self, population_size: int, notes: List[int], number_of_quarters: int,
                 notes_per_quarter: List[List[int]], chords: List[List[int]], mutation_rate=0.1,
                 generations=1000):
        self.population_size = population_size
        self.notes = notes
        self.number_of_quarters = number_of_quarters
        self.notes_per_quarter = notes_per_quarter
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.chords = chords

    def get_individual(self) -> List[List[int]]:
        chords = []
        for _ in range(self.number_of_quarters):
            chords.append(random.choice(self.chords))
        return chords

    def get_population(self) -> List[List[List[int]]]:
        population = [self.get_individual() for _ in range(self.population_size)]
        return population

    @staticmethod
    def get_chord_fitness(chord: List[int], notes_in_quart: List[int]) -> int:
        chord_ = Chord(chord)
        fitness_score = chord_.notes_in_chord(notes_in_quart)
        fitness_score += chord_.chord_notes_distance(notes_in_quart)
        fitness_score += chord_.notes_chord_octave(notes_in_quart)
        return fitness_score

    def get_individual_fitness(self, individual: List[List[int]]) -> int:
        fitness_score_all = 0
        for i in range(len(individual)):
            fitness_score = self.get_chord_fitness(individual[i], self.notes_per_quarter[i])
            if fitness_score <= 50:
                inverse_chords = Chord(individual[i]).get_inverse_chord()
                best_chord = individual[i]
                best_fitness = fitness_score
                for chord in inverse_chords:
                    fitness_score = self.get_chord_fitness(chord, self.notes_per_quarter[i])
                    if fitness_score > best_fitness:
                        best_chord = chord
                        best_fitness = fitness_score
                individual[i] = best_chord
                for j in range(len(individual[i])):
                    individual[i][j] = best_chord[j]
                fitness_score = best_fitness
            fitness_score_all += fitness_score
        return fitness_score_all

    def get_population_fitness(self, population: List[List[List[int]]]) -> List[int]:
        population_fitness = []
        for individual in population:
            population_fitness.append(self.get_individual_fitness(individual))
        return population_fitness

    @staticmethod
    def roulette_wheel_select(population_fitness: List[int]) -> int:
        shift = min(population_fitness)
        if shift < 0:
            population_fitness = [x - (shift - 5) for x in population_fitness]
        sum_fitness = sum(population_fitness)
        probabilities = [fitness / sum_fitness for fitness in population_fitness]
        return np.random.choice(range(len(population_fitness)), size=1, p=probabilities)[0]

    @staticmethod
    def two_point_crossover(parent1: List[List[int]], parent2: List[List[int]]) -> Tuple[
        List[List[int]], List[List[int]]]:
        crossover_point1 = random.randint(0, len(parent1) - 1)
        crossover_point2 = random.randint(0, len(parent1) - 1)
        while crossover_point1 == crossover_point2:
            crossover_point2 = random.randint(0, len(parent1) - 1)
        if crossover_point1 > crossover_point2:
            crossover_point1, crossover_point2 = crossover_point2, crossover_point1
        offspring1 = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[
                                                                                               crossover_point2:]
        offspring2 = parent2[:crossover_point1] + parent1[crossover_point1:crossover_point2] + parent2[
                                                                                               crossover_point2:]
        return offspring1, offspring2

    def crossover(self, population: List[List[List[int]]], population_fitness: List[int], size: int) \
            -> List[List[List[int]]]:
        offsprings = []
        for _ in range(size):
            idx1, idx2 = self.roulette_wheel_select(population_fitness), self.roulette_wheel_select(population_fitness)
            while idx1 == idx2:
                idx2 = self.roulette_wheel_select(population_fitness)
            parent1, parent2 = population[idx1], population[idx2]
            offspring1, offspring2 = self.two_point_crossover(parent1, parent2)
            offsprings.append(offspring1)
            offsprings.append(offspring2)
        return offsprings

    def mutation(self, offsprings: List[List[List[int]]]) -> List[List[List[int]]]:
        for offspring in offsprings:
            for i in range(len(offspring)):
                if np.random.random() < self.mutation_rate:
                    offspring[i] = self.get_individual()[i]
        return offsprings

    @staticmethod
    def replace_parents(population: List[List[List[int]]], population_fitness: List[int],
                        offsprings: List[List[List[int]]], offsprings_fitness: List[int], size: int) -> List[
        List[List[int]]]:
        sort_index = np.argsort(population_fitness)
        population_sorted = [[] for _ in range(len(population))]
        for i in sort_index:
            population_sorted[i] = population[i]
        sort_index = np.argsort(offsprings_fitness)
        offsprings_sorted = [[] for _ in range(len(offsprings))]
        for i in sort_index:
            offsprings_sorted[i] = offsprings[i]
        parents = population_sorted[size:]
        offsprings = offsprings_sorted[-size:]
        return [*parents, *offsprings]

    def evolution(self):
        population = self.get_population()
        population_fitness = self.get_population_fitness(population)
        for i in range(self.generations):
            offsprings = self.crossover(population, population_fitness, self.population_size // 2)
            offsprings = self.mutation(offsprings)

            offsprings_fitness = self.get_population_fitness(offsprings)
            population = self.replace_parents(population, population_fitness, offsprings, offsprings_fitness,
                                              int(self.population_size * 0.3))
            population_fitness = self.get_population_fitness(population)
        return population, population_fitness