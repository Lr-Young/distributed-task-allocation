import numpy as np
import functools

class Gene:
    order_: int
    target_id_: int
    task_type_: int
    agent_id_: int
    
    def __init__(self, order: int, target_id: int, task_type: int, agent_id: int) -> None:
        self.order_ = order
        self.target_id_ = target_id
        self.task_type_ = task_type
        self.agent_id_ = agent_id


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
    if gene1.agent_id_ == gene2.agent_id_ and gene1.order_ < gene2.agent_id_:
        return -1
    return 1


class Chromesome(list):
    target_num_: int
    target_type_num_: int
    US_list_: list[int]
    UA_list_: list[int]

    def __init__(self, target_num: int, target_type_num: int, US_list: list[int], UA_list: list[int]):
        self.genes_: list[Gene] = []
        self.target_num_ = target_num
        self.target_type_num_ = target_type_num
        self.US_list_ = US_list
        self.UA_list_ = UA_list

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

    def __setitem__(self, index, item):
        self.genes_[index] = item

    def __getitem__(self, index):
        return self.genes_[index]
    
    def __len__(self):
        return len(self.genes_)

    def sort_genes_by_order(self):
        self.genes_.sort(key=sort_key_by_order)

    def sort_genes_by_target(self):
        self.genes_.sort(key=functools.cmp_to_key(sort_cmp_by_target))

    def sort_genes_by_agent(self):
        self.genes_.sort(key=functools.cmp_to_key(sort_cmp_by_agent))

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
    


def crossover(parents: list[Chromesome], p: float) -> list[Chromesome]:
    parents_len = len(parents)
    for i in range(0, parents_len - 1, 2):
        if np.random.rand() < p:
            crossover_chromesome(parents[i], parents[i + 1])


def mutate_chromesome_order_of_tasks(parent: Chromesome):
    print(parent)
    parent.sort_genes_by_target()
    print(parent)
    target_num_list = np.arange(parent.target_num_)
    target_type_list = np.arange(parent.target_type_num_)
    target_type_num = parent.target_type_num_
    np.random.shuffle(target_num_list)
    tmp_order_list = [gene.order_ for gene in parent.genes_]
    index = 0
    for target_num in target_num_list:
        for offset in target_type_list:
            parent[index].order_ = tmp_order_list[target_num * target_type_num + offset]
            index += 1
    print(parent)
    parent.sort_genes_by_order()
    print(parent)
    a = 1




def test():
    population = population_initialization(4, 3, 4, [1, 2], [3, 4])
    mutate_chromesome_order_of_tasks(population[0])

test()
