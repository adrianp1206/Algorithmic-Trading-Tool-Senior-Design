import random
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from train_boost import train_xgboost

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
generations = 20
mutation_rate = 0.1

def fitness_function(feature_subset, X, y):
    """
    Fitness function that trains an XGBoost model on a subset of features
    and returns the accuracy as the fitness score.
    
    Args:
    feature_subset: List of str, The selected features for this individual.
    X: DataFrame, The complete feature matrix.
    y: Series, The target variable.
    
    Returns:
    float, The accuracy score of the model trained on the feature subset.
    """
    # Train the model and get accuracy
    _, accuracy = train_xgboost(X, y, feature_subset)
    return accuracy

# Initialize the population
def initialize_population():
    population = []
    for _ in range(population_size):
        # Each individual is a random subset of features
        individual = random.sample(all_features, k=random.randint(5, len(all_features)))
        population.append(individual)
    return population

# Selection: Select the top-performing individuals
def selection(population, fitness_scores):
    # Pair individuals with their fitness scores
    paired_population = list(zip(population, fitness_scores))
    # Sort by fitness score in descending order
    sorted_population = sorted(paired_population, key=lambda x: x[1], reverse=True)
    # Select the top half of the population
    selected = [individual for individual, score in sorted_population[:population_size // 2]]
    return selected

# Crossover: Combine features from two parents to create an offspring
def crossover(parent1, parent2):
    # Create a child with half features from each parent
    child = list(set(parent1[:len(parent1)//2] + parent2[len(parent2)//2:]))
    return child

# Mutation: Randomly add or remove features
def mutate(individual):
    if random.random() < mutation_rate:
        if random.random() < 0.5 and len(individual) > 1:
            # Randomly remove a feature
            individual.remove(random.choice(individual))
        else:
            # Randomly add a new feature
            feature_to_add = random.choice([f for f in all_features if f not in individual])
            individual.append(feature_to_add)
    return individual

def genetic_algorithm(X, y):
    population = initialize_population()
    
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        
        # Evaluate fitness of each individual
        fitness_scores = [fitness_function(individual, X, y) for individual in population]
        
        # Selection
        selected_individuals = selection(population, fitness_scores)
        
        # Create next generation
        next_generation = []
        
        while len(next_generation) < population_size:
            # Select two parents randomly from the selected individuals
            parent1, parent2 = random.sample(selected_individuals, 2)
            
            # Perform crossover
            child = crossover(parent1, parent2)
            
            # Perform mutation
            child = mutate(child)
            
            # Add the child to the next generation
            next_generation.append(child)
        
        # Update population with next generation
        population = next_generation
    
    # Evaluate final population and return the best individual
    fitness_scores = [fitness_function(individual, X, y) for individual in population]
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual