import numpy as np
import functools
import math
import heapq
import matplotlib.pyplot as plt
import gc

class Position:
    def __init__(self, x: float, y: float, random: bool=False) -> None:
        self.x_ = x
        self.y_ = y
        if random:
            self.x_ *= np.random.rand()
            self.y_ *= np.random.rand()
    
    def distance_to(self, pos) -> float:
        return math.sqrt((self.x_ - pos.x_) ** 2 + (self.y_ - pos.y_) ** 2)


class Gene:
    
    def __init__(self, order: int, target_id: int, task_type: int, agent_id: int) -> None:
        self.order_ = order
        self.target_id_ = target_id
        self.task_type_ = task_type
        self.agent_id_ = agent_id

    def copy(self):
        return Gene(self.order_, self.target_id_, self.task_type_, self.agent_id_)


def sort_key_by_order(gene: Gene):
    return gene.order_


def sort_cmp_by_target(gene1: Gene, gene2: Gene):
    if gene1.target_id_ < gene2.target_id_:
        return -1
    if gene1.target_id_ == gene2.target_id_ and gene1.task_type_ < gene2.task_type_:
        return -1
    return 1


def sort_cmp_by_agent(gene1: Gene, gene2: Gene):
    if gene1.agent_id_ < gene2.agent_id_:
        return -1
    if gene1.agent_id_ == gene2.agent_id_ and gene1.order_ < gene2.order_:
        return -1
    return 1


def sort_cmp_by_task(gene1: Gene, gene2: Gene):
    if gene1.task_type_ < gene2.task_type_:
        return -1
    if gene1.task_type_ == gene2.task_type_ and gene1.order_ < gene2.order_:
        return -1
    return 1


class Chromesome(list):
    target_num_: int
    target_type_num_: int
    US_list_: list[int]
    UA_list_: list[int]
    agent_positions_: list[Position]
    target_positions_: list[Position]
    agent_velocities_: list[float]


    def __init__(self, target_num: int, target_type_num: int, US_list: list[int], UA_list: list[int], 
                 agent_positions: list[Position], target_positions: list[Position], agent_velocities: list[float]):
        self.genes_: list[Gene] = []
        self.target_num_ = target_num
        self.target_type_num_ = target_type_num
        self.US_list_ = US_list
        self.UA_list_ = UA_list
        self.agent_positions_ = agent_positions
        self.target_positions_ = target_positions
        self.agent_velocities_ = agent_velocities

        order_list = np.arange(target_num * target_type_num)
        target_id_list = [val for val in range(target_num) for _ in range(target_type_num)]
        task_type_list = [val for _ in range(target_num) for val in range(target_type_num)]
        np.random.shuffle(order_list)
        for index, order in enumerate(order_list):
            agent_id = np.random.choice(US_list)
            if task_type_list[index] == 1:
                agent_id = np.random.choice(UA_list)
            self.genes_.append(Gene(order=order, target_id=target_id_list[index], task_type=task_type_list[index], agent_id=agent_id))
        self.sort_genes_by_order()

    def fitness(self) -> float:
        self.sort_genes_by_agent()
        times = []
        time = self.time_from_agent_to_target(self[0].agent_id_, self[0].target_id_)
        for i in range(1, len(self)):
            if self[i - 1].agent_id_ != self[i].agent_id_:
                times.append(time)
                time = self.time_from_agent_to_target(self[i].agent_id_, self[i].target_id_)
            else:
                time += self.time_from_target_to_target(self[i].agent_id_, self[i - 1].target_id_, self[i].target_id_)
        return max(times)


    def time_from_agent_to_target(self, agent_id: int, target_id: int) -> float:
        return self.agent_positions_[agent_id].distance_to(self.target_positions_[target_id]) \
            / self.agent_velocities_[agent_id]
    
    def time_from_target_to_target(self, agent_id: int, target1_id: int, target2_id: int) -> float:
        return self.target_positions_[target1_id].distance_to(self.target_positions_[target2_id]) \
            / self.agent_velocities_[agent_id]
    
    def copy(self):
        ans: Chromesome = Chromesome(self.target_num_, self.target_type_num_, self.US_list_, self.UA_list_, 
             self.agent_positions_, self.target_positions_, self.agent_velocities_)
        del ans.genes_[:]
        for gene in self.genes_:
            ans.append(gene.copy())
        return ans
    
    def get_gene_set_by_agent(self) -> list[list[Gene]]:
        res: list[list[Gene]] = []
        self.sort_genes_by_agent()
        res.append([self[0]])
        for i in range(1, len(self)):
            if(self[i - 1].agent_id_ == self[i].agent_id_):
                res[len(res) - 1].append(self[i])
            else:
                res.append([self[i]])
        self.sort_genes_by_order()
        return res

    def __setitem__(self, index, item):
        self.genes_[index] = item

    def __getitem__(self, index) -> Gene:
        return self.genes_[index]
    
    def __len__(self):
        return len(self.genes_)
    
    def append(self, item: Gene):
        if isinstance(item, Gene):
            self.genes_.append(item)
        else:
            raise TypeError()

    def sort_genes_by_order(self):
        self.genes_.sort(key=sort_key_by_order)

    def sort_genes_by_target(self):
        self.genes_.sort(key=functools.cmp_to_key(sort_cmp_by_target))

    def sort_genes_by_agent(self):
        self.genes_.sort(key=functools.cmp_to_key(sort_cmp_by_agent))

    def sort_genes_by_task(self):
        self.genes_.sort(key=functools.cmp_to_key(sort_cmp_by_task))
        

    def __str__(self):
        order_str =  "order:  "
        target_str = "target: "
        type_str =   "type:   "
        agent_str =  "agent:  "
        for gene in self.genes_:
            order_str += str(gene.order_) + " "
            target_str += str(gene.target_id_) + " "
            type_str += str(gene.task_type_) + " "
            agent_str += str(gene.agent_id_) + " "
        return "Chromesome\n" + order_str + "\n" + target_str + "\n" + type_str + "\n" + agent_str + "\n"
    


def population_initialization(target_num: int, target_type_num: int, population_size: int, US_list: list[int], UA_list: list[int], 
                              agent_positions: list[Position], target_positions: list[Position], agent_velocities: list[float]) -> list[Chromesome]:
    population: list[Chromesome] = []
    for _ in range(population_size):
        population.append(Chromesome(target_num=target_num, target_type_num=target_type_num, US_list=US_list, UA_list=UA_list, 
                                     agent_positions=agent_positions, target_positions=target_positions, agent_velocities=agent_velocities))
    return population


def crossover_chromesome(father: Chromesome, mother: Chromesome) -> list[Chromesome]:
    child1 = father.copy()
    child2 = mother.copy()
    child1.sort_genes_by_target()
    child2.sort_genes_by_target()

    gene_num = len(child1)
    [crossover_sites1, crossover_sites2] = np.random.choice(np.arange(gene_num + 1), 2, replace=False)
    if crossover_sites1 > crossover_sites2:
        tmp = crossover_sites1
        crossover_sites1 = crossover_sites2
        crossover_sites2 = tmp
    tmp_agent_id_list = [gene.agent_id_ for gene in child1.genes_]
    for index in range(crossover_sites1, crossover_sites2):
        child1[index].agent_id_ = child2[index].agent_id_
        child2[index].agent_id_ = tmp_agent_id_list[index]

    child1.sort_genes_by_order()
    child2.sort_genes_by_order()

    return [child1, child2]


def mutate_chromesome_order_of_tasks(parent: Chromesome) -> Chromesome:
    child = parent.copy()
    child.sort_genes_by_target()
    target_index_list = np.arange(child.target_num_)
    target_type_list = np.arange(child.target_type_num_)
    target_type_num = child.target_type_num_
    np.random.shuffle(target_index_list)
    tmp_order_list = [gene.order_ for gene in child.genes_]
    for index, target_index in enumerate(target_index_list):
        for offset in target_type_list:
            child[index].order_ = tmp_order_list[target_index * target_type_num + offset]
    child.sort_genes_by_order()
    return child


def mutate_chromesome_task(parent: Chromesome) -> Chromesome:
    child = parent.copy()
    child.sort_genes_by_task()
    target_num = child.target_num_
    mutation_site = np.random.choice(child.target_type_num_)
    in_target_agent_index_list = np.arange(target_num)
    np.random.shuffle(in_target_agent_index_list)
    tmp_agent_list = [child.genes_[mutation_site * target_num + i].agent_id_ for i in range(target_num)]
    for index, agent_index in enumerate(in_target_agent_index_list):
        child[mutation_site * target_num + index].agent_id_ = tmp_agent_list[agent_index]
    child.sort_genes_by_order()
    return child
    

def mutate_single_gene_agent(parent: Chromesome) -> Chromesome:
    child = parent.copy()
    mutation_site = np.random.choice(len(child))
    if child[mutation_site].task_type_ == 1:
        tmp_agent_list = [agent_id for agent_id in child.UA_list_]
    else:
        tmp_agent_list = [agent_id for agent_id in child.US_list_]
    tmp_agent_list.remove(child[mutation_site].agent_id_)
    child[mutation_site].agent_id_ = np.random.choice(tmp_agent_list)
    return child


def crossover(parents: list[Chromesome], children_num: int) -> list[Chromesome]:
    selected_parent_index = np.random.choice(np.arange(len(parents)), children_num, replace=False)
    parents = [parents[i] for i in selected_parent_index]
    parents_len = len(parents)
    parity = parents_len % 2 == 1
    children: list[Chromesome] = []
    for i in range(0, parents_len - 1, 2):
        children.extend(crossover_chromesome(parents[i], parents[i + 1]))
    if parity:
        children.append(parents[len(parents) - 1].copy())
    return children


def mutate(parents: list[Chromesome], children_num: int) -> list[Chromesome]:
    parent = parents[np.random.choice(len(parents))]  # randomly select one parent chromesome
    children: list[Chromesome] = []
    for _ in range(children_num):
        mutation_type = np.random.choice(3)
        if mutation_type == 0:
            children.append(mutate_single_gene_agent(parent))
        elif mutation_type == 1:
            children.append(mutate_chromesome_task(parent))
        elif mutation_type == 2:
            children.append(mutate_chromesome_order_of_tasks(parent))
    return children


def select(population: list[Chromesome]) -> list[Chromesome]:
    # Double-pointer method to optimize the time complexity to nlogn.
    # Fitness values are linearly assigned between zero and one to the chromosomes of
    # each generation based on a minimization objective.
    child_population: list[Chromesome] = []
    sorted_indices = sorted(range(len(population)), key=lambda index : population[index].fitness())
    step = 1 / len(population)
    fitness = np.arange(step, 1 + step, step)
    fitness_probability = fitness / sum(fitness)
    fitness_probability = np.cumsum(fitness_probability)
    random_probability = np.sort(np.random.rand(len(population)))
    fitness_index = 0
    random_index = 0
    while fitness_index < len(fitness_probability) and random_index < len(random_probability):
        if random_probability[random_index] < fitness_probability[fitness_index]:
            child_population.append(population[sorted_indices[fitness_index]].copy())
            random_index += 1
        else:
            fitness_index += 1
    del population[:]
    return child_population


def elite_select(population: list[Chromesome], elite_num: int) -> list[Chromesome]:
    elites = heapq.nsmallest(elite_num, population, lambda individual : individual.fitness())
    return [elite.copy() for elite in elites]


def adaptive_evolve(target_num: int, target_type_num: int, population_size: int, elite_num: int, iteration_num: int, 
                    US_list: list[int], UA_list: list[int], agent_positions: list[Position],target_positions: list[Position], 
                    agent_velocities: list[float]) -> tuple[Chromesome, list[float], list[float]]:
    population = population_initialization(target_num, target_type_num, population_size, 
                                           US_list, UA_list, agent_positions, target_positions, agent_velocities)
    avg_fitness_list = []
    best_fitness_list = []
    for iteration in range(iteration_num):
        crossover_num = round((population_size - elite_num) * math.exp(-iteration / iteration_num))
        mutation_num = population_size - elite_num - crossover_num

        elite_parents = elite_select(population, elite_num)
        population = select(population)
        crossover_offspring = crossover(population, crossover_num)
        mutation_offspring = mutate(population, mutation_num)
        del population[:]
        population.extend(elite_parents)
        population.extend(crossover_offspring)
        population.extend(mutation_offspring)

        avg_fitness_list.append(sum([individual.fitness() for individual in population]) / population_size)
        best_fitness_list.append(min([individual.fitness() for individual in population]))
        gc.collect()
    best_solution = min(population, key=lambda individual : individual.fitness())
    return best_solution, avg_fitness_list, best_fitness_list


def draw(solution: Chromesome, agent_positions: list[Position], target_positions: list[Position], agent_velocities: list[float], 
         avg_fitness_list: list[float], best_fitness_list: list[float]) -> None:
    x = range(len(avg_fitness_list))
    plt.subplot(2, 2, 1)
    plt.plot(x, avg_fitness_list)
    plt.subplot(2, 2, 3)
    plt.plot(x, best_fitness_list)
    plt.subplot(1, 2, 2)
    colors: list[tuple[float, float, float]] = [tuple([np.random.rand() for _ in range(3)]) for _ in range(len(agent_positions))]
    target_x = []
    target_y = []
    for position in target_positions:
        target_x.append(position.x_)
        target_y.append(position.y_)
    plt.scatter(target_x, target_y, c="black", marker='x')
    for i, position in enumerate(agent_positions):
        plt.scatter(position.x_, position.y_, c=colors[i])
    for genes in solution.get_gene_set_by_agent():
        agent_id = genes[0].agent_id_
        trajectory_x = [agent_positions[agent_id].x_]
        trajectory_y = [agent_positions[agent_id].y_]
        trajectory_x.extend([target_positions[gene.target_id_].x_ for gene in genes])
        trajectory_y.extend([target_positions[gene.target_id_].y_ for gene in genes])
        distance = 0
        for j in range(1, len(trajectory_x)):
            distance += math.sqrt((trajectory_x[j] - trajectory_x[j - 1]) ** 2 + (trajectory_y[j] - trajectory_y[j - 1]) ** 2)
        time = distance / agent_velocities[agent_id]
        # plt.legend()
        plt.plot(trajectory_x, trajectory_y, label=f'Agent <{agent_id}, {agent_velocities[agent_id]}>: <{distance}, {time}>', c=colors[agent_id])

    plt.show()

def test():
    target_num = 15
    population_size = 100
    elite_num = 4
    iteration_num = 300
    area_length = 10000
    agent_positions: list[Position] = []
    target_positions: list[Position] = []
    agent_velocities: list[float] = []
    agent_positions.append(Position(0, 0))  # agent_id: 0
    agent_positions.append(Position(0, area_length))
    agent_positions.append(Position(area_length, area_length))
    agent_positions.append(Position(area_length, 0))
    agent_positions.append(Position(0, area_length / 2))
    agent_positions.append(Position(area_length / 2, area_length))
    agent_positions.append(Position(area_length, area_length / 2))
    agent_positions.append(Position(area_length / 2, 0))  # agent_id: 7
    agent_velocities.append(70)
    agent_velocities.append(80)
    agent_velocities.append(90)
    agent_velocities.append(100)
    agent_velocities.append(60)
    agent_velocities.append(70)
    agent_velocities.append(80)
    agent_velocities.append(90)
    for _ in range(target_num):
        target_positions.append(Position(area_length, area_length, True))
    solution, avg_fitness_list, best_fitness_list = adaptive_evolve(target_num, 3, population_size, elite_num, iteration_num,
                                 [0, 1, 2, 3], [4, 5, 6, 7], agent_positions, target_positions, agent_velocities)
    draw(solution, agent_positions, target_positions, agent_velocities, avg_fitness_list, best_fitness_list)

