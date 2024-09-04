import random
import math
import numpy as np
from deap import base, creator, tools, algorithms

# Parameters
N = 10  # Number of targets
M = 6  # Number of agents
L = 100  # Size of the area (L km * L km)
num_us = 2  # Number of US agents
num_uc = 2  # Number of UC agents
num_um = 2  # Number of UM agents

# Random positions for targets and agents
np.random.seed(0)
targets_positions = np.random.uniform(0, L, (N, 2))
agents_positions = np.random.uniform(0, L, (M, 2))
agents_velocities = np.random.uniform(5, 10, M)

# Agents types
agents_types = ["US", "US", "UC", "UC", "UM", "UM"]

# Task types and constraints
task_types = ["C", "A", "V"]  # 3 tasks per target
tasks_per_target = len(task_types)
total_tasks = N * tasks_per_target

# Genetic algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Create toolbox for the genetic algorithm
toolbox = base.Toolbox()

# Define the individual creation function (random assignment of tasks to agents)
def create_individual():
    # Assign tasks to agents randomly, respecting agent-task constraints
    tasks = []
    for i in range(N):
        for task_type in task_types:
            valid_agents = [idx for idx, agent_type in enumerate(agents_types) 
                            if (task_type == "C" and agent_type in ["US", "UC"]) or 
                               (task_type == "A" and agent_type in ["UC", "UM"]) or 
                               (task_type == "V" and agent_type in ["US", "UC"])]
            tasks.append(random.choice(valid_agents))
    return creator.Individual(tasks)

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness function
def fitness(individual):
    # Calculate the total distance each agent must travel to complete the assigned tasks
    # and the corresponding time to complete them
    agent_paths = {i: [] for i in range(M)}
    agent_times = {i: 0.0 for i in range(M)}
    
    # Assign tasks to agents
    for idx, agent_idx in enumerate(individual):
        target_idx = idx // tasks_per_target
        agent_paths[agent_idx].append(targets_positions[target_idx])

    # Calculate the distance traveled by each agent and the time to complete the assigned tasks
    for agent_idx, path in agent_paths.items():
        if not path:
            continue
        
        # Calculate the total distance for the agent's path
        position = agents_positions[agent_idx]
        total_distance = 0
        for task_position in path:
            distance = math.sqrt((task_position[0] - position[0]) ** 2 + 
                                 (task_position[1] - position[1]) ** 2)
            total_distance += distance
            position = task_position
        
        agent_times[agent_idx] = total_distance / agents_velocities[agent_idx]
    
    # The fitness value is the maximum time among all agents
    return (max(agent_times.values()),)

toolbox.register("evaluate", fitness)

# Selection, crossover, and mutation
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=0.05)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

# Genetic algorithm parameters
population_size = 100
crossover_prob = 0.7
mutation_prob = 0.2
generations = 50

# Create initial population
population = toolbox.population(n=population_size)

# Run the genetic algorithm
algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=generations, 
                    stats=None, halloffame=None, verbose=True)

# Get the best solution
best_individual = tools.selBest(population, 1)[0]
best_fitness = best_individual.fitness.values[0]

print("Best fitness:", best_fitness)
print("Best task allocation:", best_individual)
