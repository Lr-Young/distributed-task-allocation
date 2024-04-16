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
        return 1
    if gene1.target_id_ == gene2.target_id_ and gene1.task_type_ < gene2.task_type_:
        return 1
    return -1


def sort_cmp_by_agent(gene1: Gene, gene2: Gene):
    if gene1.agent_id_ < gene2.agent_id_:
        return 1
    if gene1.agent_id_ == gene2.agent_id_ and gene1.order_ < gene2.agent_id_:
        return 1
    return -1


class Chromesome:
    genes_: list[Gene] = []

    def __init__(self, target_num: int, target_type_num: int, US_list: list[int], UA_list: list[int]):
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

    def sort_genes_by_order(self):
        self.genes_.sort(key=sort_key_by_order)

    def sort_genes_by_target(self):
        self.genes_.sort(key=functools.cmp_to_key(sort_cmp_by_target))

    def sort_genes_by_agent(self):
        self.genes_.sort(key=functools.cmp_to_key(sort_cmp_by_agent))
    


def population_initialization(target_num: int, target_type_num: int, population_size: int, US_list: list[int], UA_list: list[int]) -> list[Chromesome]:
    population: list[Chromesome] = []
    for _ in range(population_size):
        population.append(Chromesome(target_num=target_num, target_type_num=target_type_num, US_list=US_list, UA_list=UA_list))
    return population

