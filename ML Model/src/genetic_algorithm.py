import random
import numpy as np
import xgboost as xgb
from train_boost import train_xgboost_with_timeseries_cv

# List of all technical indicators as features
all_features = [
    'BB_upper', 'BB_middle', 'BB_lower', 'DEMA', 'MIDPOINT', 'MIDPRICE', 'SMA', 'T3', 
    'TEMA', 'TRIMA', 'WMA', 'ADX', 'ADXR', 'APO', 'AROON_DOWN', 'AROON_UP', 'AROONOSC', 
    'CCI', 'CMO', 'MACD', 'MACD_signal', 'MACD_hist', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 
    'PLUS_DI', 'PLUS_DM', 'ROC', 'RSI', 'STOCH_slowk', 'STOCH_slowd', 'STOCH_fastk', 
    'STOCH_fastd', 'ATR', 'NATR', 'TRANGE', 'AD', 'ADOSC', 'OBV', 'AVGPRICE', 'MEDPRICE', 
    'TYPPRICE', 'WCLPRICE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR_inphase', 
    'HT_PHASOR_quadrature', 'HT_SINE', 'HT_LEADSINE', 'HT_TRENDMODE'
]

# Parameters for the GA
population_size = 10
generations = 50
mutation_rate = 0.1

# Fitness function
def fitness_function(feature_subset, X, y):
    accuracy, _ = train_xgboost_with_timeseries_cv(X, y, feature_subset)  # Adjust if needed in train_xgboost_with_timeseries_cv
    return accuracy  # Only return the accuracy score as fitness

# Initialize population
def initialize_population():
    population = []
    for _ in range(population_size):
        individual = random.sample(all_features, k=random.randint(5, len(all_features)))
        population.append(individual)
    return population

# Selection
def selection(population, fitness_scores):
    paired_population = list(zip(population, fitness_scores))
    sorted_population = sorted(paired_population, key=lambda x: x[1], reverse=True)
    selected = [individual for individual, score in sorted_population[:population_size // 2]]
    return selected

# Crossover
def crossover(parent1, parent2):
    child = list(set(parent1[:len(parent1)//2] + parent2[len(parent2)//2:]))
    return child

# Mutation
def mutate(individual):
    if random.random() < mutation_rate:
        if random.random() < 0.5 and len(individual) > 1:
            individual.remove(random.choice(individual))
        else:
            feature_to_add = random.choice([f for f in all_features if f not in individual])
            individual.append(feature_to_add)
    return individual

# Genetic Algorithm
def genetic_algorithm(X, y):
    population = initialize_population()
    
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        
        fitness_scores = [fitness_function(individual, X, y) for individual in population]
        
        selected_individuals = selection(population, fitness_scores)
        
        next_generation = []
        
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected_individuals, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            
            # Ensure unique individuals in the next generation
            if child not in next_generation:
                next_generation.append(child)
        
        population = next_generation
    
    # Final evaluation to get the best individual
    fitness_scores = [fitness_function(individual, X, y) for individual in population]
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual
