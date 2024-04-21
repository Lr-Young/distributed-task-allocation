import numpy as np
import functools
import math


class Position:
    def __init__(self, x: float, y: float) -> None:
        self.x_ = 0.0
        self.y_ = 0.0
    
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
                time = 0.0
            time += self.time_from_target_to_target(self[i].agent_id_, self[i - 1].target_id_, self[i].target_id_)
        return max(times)


    def time_from_agent_to_target(self, agent_id: int, target_id: int) -> float:
        return self.agent_positions_[agent_id].distance_to(self.target_positions_[target_id]) \
            / self.agent_velocities_[agent_id]
    
    def time_from_target_to_target(self, agent_id: int, target1_id: int, target2_id: int) -> float:
        return self.agent_positions_[target1_id].distance_to(self.target_positions_[target2_id]) \
            / self.agent_velocities_[agent_id]
    
    def copy(self):
        ans: Chromesome = Chromesome(self.target_num_, self.target_type_num_, self.US_list_, self.UA_list_, 
             self.agent_positions_, self.target_positions_, self.agent_velocities_)
        del ans.genes_[:]
        for gene in self.genes_:
            ans.append(gene.copy())
        return ans
        

    def __setitem__(self, index, item):
        self.genes_[index] = item

    def __getitem__(self, index) -> Gene:
        return self.genes_[index]
    
    def __len__(self):
        return len(self.genes_)

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
    


def population_initialization(target_num: int, target_type_num: int, population_size: int, US_list: list[int], UA_list: list[int]) -> list[Chromesome]:
    population: list[Chromesome] = []
    for _ in range(population_size):
        population.append(Chromesome(target_num=target_num, target_type_num=target_type_num, US_list=US_list, UA_list=UA_list))
    return population


def crossover_chromesome(father: Chromesome, mother: Chromesome):
    father.sort_genes_by_target()
    mother.sort_genes_by_target()

    gene_num = len(father)
    [crossover_sites1, crossover_sites2] = np.random.choice(np.arange(gene_num + 1), 2, replace=False)
    if crossover_sites1 > crossover_sites2:
        tmp = crossover_sites1
        crossover_sites1 = crossover_sites2
        crossover_sites2 = tmp
    tmp_agent_id_list = [gene.agent_id_ for gene in father.genes_]
    for index in range(crossover_sites1, crossover_sites2):
        father[index].agent_id_ = mother[index].agent_id_
        mother[index].agent_id_ = tmp_agent_id_list[index]

    father.sort_genes_by_order()
    mother.sort_genes_by_order()


def mutate_chromesome_order_of_tasks(parent: Chromesome):
    parent.sort_genes_by_target()
    target_index_list = np.arange(parent.target_num_)
    target_type_list = np.arange(parent.target_type_num_)
    target_type_num = parent.target_type_num_
    np.random.shuffle(target_index_list)
    tmp_order_list = [gene.order_ for gene in parent.genes_]
    for index, target_index in enumerate(target_index_list):
        for offset in target_type_list:
            parent[index].order_ = tmp_order_list[target_index * target_type_num + offset]
    parent.sort_genes_by_order()


def mutate_chromesome_task(parent: Chromesome):
    parent.sort_genes_by_task()
    target_num = parent.target_num_
    mutation_site = np.random.choice(parent.target_type_num_)
    in_target_agent_index_list = np.arange(target_num)
    np.random.shuffle(in_target_agent_index_list)
    tmp_agent_list = [parent.genes_[mutation_site * target_num + i].agent_id_ for i in range(target_num)]
    for index, agent_index in enumerate(in_target_agent_index_list):
        parent[mutation_site * target_num + index].agent_id_ = tmp_agent_list[agent_index]
    parent.sort_genes_by_order()
    

def mutate_single_gene_agent(parent: Chromesome):
    print(parent)
    mutation_site = np.random.choice(len(parent))
    print(mutation_site)
    if parent[mutation_site].task_type_ == 1:
        tmp_agent_list = [agent_id for agent_id in parent.UA_list_]
    else:
        tmp_agent_list = [agent_id for agent_id in parent.US_list_]
    tmp_agent_list.remove(parent[mutation_site].agent_id_)
    print(tmp_agent_list)
    print("US list " + str(parent.US_list_))
    print("UA list " + str(parent.UA_list_))
    parent[mutation_site].agent_id_ = np.random.choice(tmp_agent_list)
    print(parent)


def crossover(parents: list[Chromesome], children_num: int) -> list[Chromesome]:
    parents_len = len(parents)
    children: list[Chromesome] = []
    for i in range(0, parents_len - 1, 2):
        crossover_chromesome(parents[i], parents[i + 1])
        children.extend([parents[i], parents[i + 1]])
        if len(children) >= children_num:
            break
    del parents[:]  # to avoid memory leaking
    return children


def mutate(parents: list[Chromesome], children_num: int) -> list[Chromesome]:
    parent = np.random.choice(parents)  # randomly select one parent chromesome
    children: list[Chromesome] = []
    for _ in range(children_num):
        mutation_type = np.random.choice(3)
        if mutation_type == 0:
            mutate_single_gene_agent(parent)
        elif mutation_type == 1:
            mutate_chromesome_task(parent)
        elif mutation_type == 2:
            mutate_chromesome_order_of_tasks(parent)
        children.append(parent)
    del parents[:]  # to avoid memory leaking
    return children


def select(population: list[Chromesome]) -> list[Chromesome]:
    # double-pointer method to optimize the time complexity to nlogn
    child_population: list[Chromesome] = []
    fitness = [individual.fitness() for individual in population]
    fitness /= sum(fitness)
    fitness = np.cumsum(fitness)


def test():
    population = population_initialization(5, 3, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9], [11, 12, 13, 14, 15, 16, 17, 18, 19])
    mutate_single_gene_agent(population[0])


