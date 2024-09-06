import numpy as np
import functools
import math
import heapq
import matplotlib.pyplot as plt
import gc

class Position:
    def __init__(self, x: float, y: float, random: bool=False) -> None:
        """
        初始化一个Position对象。
        
        参数:
        x (float): x坐标值。
        y (float): y坐标值。
        random (bool): 是否随机化坐标值。如果为True，则x和y将乘以一个随机生成的0到1之间的数。
        """
        self.x_ = x
        self.y_ = y
        if random:
            # 如果random为True，则将x和y乘以一个随机数，使坐标值随机化
            self.x_ *= np.random.rand()
            self.y_ *= np.random.rand()
    
    def distance_to(self, pos) -> float:
        """
        计算当前Position对象与另一个Position对象之间的欧几里得距离。
        
        参数:
        pos (Position): 另一个Position对象。
        
        返回:
        float: 两个Position对象之间的距离。
        """
        return math.sqrt((self.x_ - pos.x_) ** 2 + (self.y_ - pos.y_) ** 2)

class Gene:
    """
    Gene 类表示具有特定属性的基因，如顺序、目标ID、任务类型和智能体（机器人）ID。
    """

    def __init__(self, order: int, target_id: int, task_type: int, agent_id: int) -> None:
        """
        使用给定的参数初始化一个 Gene 对象。

        :param order: 表示基因顺序的整数。
        :param target_id: 表示与基因相关的目标ID的整数。
        :param task_type: 表示基因相关任务类型的整数。
        :param agent_id: 表示与基因相关的智能体（机器人）ID的整数。
        """
        self.order_ = order           # 存储基因的顺序
        self.target_id_ = target_id   # 存储与基因相关的目标ID
        self.task_type_ = task_type   # 存储与基因相关的任务类型
        self.agent_id_ = agent_id     # 存储与基因相关的智能体（机器人）ID

    def copy(self):
        """
        创建当前 Gene 对象的副本。

        :return: 一个具有与当前 Gene 对象相同属性的新 Gene 对象。
        """
        return Gene(self.order_, self.target_id_, self.task_type_, self.agent_id_)


def sort_key_by_order(gene: Gene):
    """
    根据基因的 order_ 属性生成排序键。

    参数:
    gene (Gene): Gene 类的实例。

    返回:
    int: 基因的 order_ 属性值。
    """
    return gene.order_


def sort_cmp_by_target(gene1: Gene, gene2: Gene):
    """
    比较两个基因对象的 target_id_ 和 task_type_ 属性，用于排序。
    
    参数:
    gene1 (Gene): 第一个 Gene 类的实例。
    gene2 (Gene): 第二个 Gene 类的实例。

    返回:
    int: 如果 gene1 应该排在 gene2 前面，返回 -1；如果 gene1 和 gene2 相等，返回 -1；否则返回 1。
    """
    if gene1.target_id_ < gene2.target_id_:
        return -1
    if gene1.target_id_ == gene2.target_id_ and gene1.task_type_ < gene2.task_type_:
        return -1
    return 1


def sort_cmp_by_agent(gene1: Gene, gene2: Gene):
    """
    比较两个基因对象的 agent_id_ 和 order_ 属性，用于排序。
    
    参数:
    gene1 (Gene): 第一个 Gene 类的实例。
    gene2 (Gene): 第二个 Gene 类的实例。

    返回:
    int: 如果 gene1 应该排在 gene2 前面，返回 -1；如果 gene1 和 gene2 相等，返回 -1；否则返回 1。
    """
    if gene1.agent_id_ < gene2.agent_id_:
        return -1
    if gene1.agent_id_ == gene2.agent_id_ and gene1.order_ < gene2.order_:
        return -1
    return 1


def sort_cmp_by_task(gene1: Gene, gene2: Gene):
    """
    比较两个基因对象的 task_type_ 和 order_ 属性，用于排序。
    
    参数:
    gene1 (Gene): 第一个 Gene 类的实例。
    gene2 (Gene): 第二个 Gene 类的实例。

    返回:
    int: 如果 gene1 应该排在 gene2 前面，返回 -1；如果 gene1 和 gene2 相等，返回 -1；否则返回 1。
    """
    if gene1.task_type_ < gene2.task_type_:
        return -1
    if gene1.task_type_ == gene2.task_type_ and gene1.order_ < gene2.order_:
        return -1
    return 1

class Chromesome(list):
    # 定义类的成员变量
    target_num_: int  # 目标数量
    target_type_num_: int  # 目标类型数量
    US_list_: list[int]  # US类型智能体（机器人）的列表
    UA_list_: list[int]  # UA类型智能体（机器人）的列表
    agent_positions_: list[Position]  # 智能体（机器人）的位置列表
    target_positions_: list[Position]  # 目标的位置列表
    agent_velocities_: list[float]  # 智能体（机器人）的速度列表

    # 初始化方法，构造函数
    def __init__(self, target_num: int, target_type_num: int, US_list: list[int], UA_list: list[int],
                 agent_positions: list[Position], target_positions: list[Position], agent_velocities: list[float]):
        self.genes_: list[Gene] = []  # 基因列表
        self.target_num_ = target_num  # 初始化目标数量
        self.target_type_num_ = target_type_num  # 初始化目标类型数量
        self.US_list_ = US_list  # 初始化US智能体（机器人）列表
        self.UA_list_ = UA_list  # 初始化UA智能体（机器人）列表
        self.agent_positions_ = agent_positions  # 初始化智能体（机器人）位置列表
        self.target_positions_ = target_positions  # 初始化目标位置列表
        self.agent_velocities_ = agent_velocities  # 初始化智能体（机器人）速度列表

        # 创建顺序列表
        order_list = np.arange(target_num * target_type_num)
        # 创建目标ID列表
        target_id_list = [val for val in range(target_num) for _ in range(target_type_num)]
        # 创建任务类型列表
        task_type_list = [val for _ in range(target_num) for val in range(target_type_num)]
        # 随机打乱顺序列表
        np.random.shuffle(order_list)
        # 根据顺序列表生成基因
        for index, order in enumerate(order_list):
            agent_id = np.random.choice(US_list)  # 随机选择US智能体（机器人）
            if task_type_list[index] == 1:
                agent_id = np.random.choice(UA_list)  # 如果任务类型为1，随机选择UA智能体（机器人）
            self.genes_.append(Gene(order=order, target_id=target_id_list[index], task_type=task_type_list[index], agent_id=agent_id))
        self.sort_genes_by_order()  # 对基因按顺序排序

    # 计算适应度函数
    def fitness(self) -> float:
        times = []  # 存储每个智能体（机器人）完成任务所需时间
        gene_sets = self.get_gene_set_by_agent()  # 获取按智能体（机器人）分组的基因集合
        for genes in gene_sets:
            # 从智能体（机器人）到第一个目标的时间
            time = self.time_from_agent_to_target(genes[0].agent_id_, genes[0].target_id_)
            for i in range(1, len(genes)):
                # 从一个目标到下一个目标的时间
                time += self.time_from_target_to_target(genes[0].agent_id_, genes[i - 1].target_id_, genes[i].target_id_)
            times.append(time)
        return max(times)  # 返回最大时间作为适应度

    # 计算智能体（机器人）到目标的时间
    def time_from_agent_to_target(self, agent_id: int, target_id: int) -> float:
        return self.agent_positions_[agent_id].distance_to(self.target_positions_[target_id]) \
            / self.agent_velocities_[agent_id]
    
    # 计算目标到目标的时间
    def time_from_target_to_target(self, agent_id: int, target1_id: int, target2_id: int) -> float:
        return self.target_positions_[target1_id].distance_to(self.target_positions_[target2_id]) \
            / self.agent_velocities_[agent_id]
    
    # 复制当前染色体
    def copy(self):
        ans: Chromesome = Chromesome(self.target_num_, self.target_type_num_, self.US_list_, self.UA_list_,
             self.agent_positions_, self.target_positions_, self.agent_velocities_)
        del ans.genes_[:]  # 清空目标基因列表
        for gene in self.genes_:
            ans.append(gene.copy())  # 复制每个基因
        return ans  # 返回复制的染色体

    def get_gene_set_by_agent(self) -> list[list[Gene]]:
        """
        获取按智能体（机器人）分类的基因集合。
        首先按智能体（机器人）对基因进行排序，然后将基因按智能体（机器人）分组。
        返回一个包含多个基因列表的列表，每个子列表对应一个智能体（机器人）的基因。
        """
        res: list[list[Gene]] = []  # 初始化结果列表
        self.sort_genes_by_agent()  # 按智能体（机器人）对基因进行排序
        res.append([self[0]])  # 将第一个基因添加到结果列表的第一个子列表中
        for i in range(1, len(self)):
            if(self[i - 1].agent_id_ == self[i].agent_id_):  # 如果前一个基因和当前基因属于同一个智能体（机器人）
                res[len(res) - 1].append(self[i])  # 将当前基因添加到当前子列表中
            else:
                res.append([self[i]])  # 否则，创建一个新的子列表，并将当前基因添加到其中
        self.sort_genes_by_order()  # 按顺序重新排序基因
        return res  # 返回按智能体（机器人）分组的基因集合

    def __setitem__(self, index, item):
        """
        设置指定索引处的基因。
        """
        self.genes_[index] = item

    def __getitem__(self, index) -> Gene:
        """
        获取指定索引处的基因。
        """
        return self.genes_[index]
    
    def __len__(self):
        """
        获取基因列表的长度。
        """
        return len(self.genes_)
    
    def append(self, item: Gene):
        """
        向基因列表末尾添加一个基因。
        如果item不是Gene类型，则抛出TypeError。
        """
        if isinstance(item, Gene):
            self.genes_.append(item)
        else:
            raise TypeError()

    def sort_genes_by_order(self):
        """
        按顺序对基因进行排序。
        """
        self.genes_.sort(key=sort_key_by_order)

    def sort_genes_by_target(self):
        """
        按目标对基因进行排序。
        """
        self.genes_.sort(key=functools.cmp_to_key(sort_cmp_by_target))

    def sort_genes_by_agent(self):
        """
        按智能体（机器人）对基因进行排序。
        """
        self.genes_.sort(key=functools.cmp_to_key(sort_cmp_by_agent))

    def sort_genes_by_task(self):
        """
        按任务对基因进行排序。
        """
        self.genes_.sort(key=functools.cmp_to_key(sort_cmp_by_task))

    def __str__(self):
        # 初始化字符串变量，用于存储基因的不同属性
        order_str =  "order:  "  # 存储基因的顺序
        target_str = "target: "  # 存储基因的目标ID
        type_str =   "type:   "  # 存储基因的任务类型
        agent_str =  "agent:  "  # 存储基因的智能体（机器人）ID

        # 遍历所有基因，将每个基因的属性添加到相应的字符串中
        for gene in self.genes_:
            order_str += str(gene.order_) + " "       # 添加基因的顺序到order_str
            target_str += str(gene.target_id_) + " "  # 添加基因的目标ID到target_str
            type_str += str(gene.task_type_) + " "    # 添加基因的任务类型到type_str
            agent_str += str(gene.agent_id_) + " "    # 添加基因的智能体（机器人）ID到agent_str

        # 返回格式化后的字符串，包含所有基因的详细信息
        return "Chromesome\n" + order_str + "\n" + target_str + "\n" + type_str + "\n" + agent_str + "\n"


def population_initialization(target_num: int, target_type_num: int, population_size: int, US_list: list[int], UA_list: list[int], 
                              agent_positions: list[Position], target_positions: list[Position], agent_velocities: list[float]) -> list[Chromesome]:
    """
    初始化种群。
    
    参数:
    - target_num (int): 目标数量。
    - target_type_num (int): 目标类型数量。
    - population_size (int): 种群大小。
    - US_list (list[int]): US类型智能体（机器人）列表。
    - UA_list (list[int]): UA类型智能体（机器人）列表。
    - agent_positions (list[Position]): 智能体（机器人）位置列表。
    - target_positions (list[Position]): 目标位置列表。
    - agent_velocities (list[float]): 智能体（机器人）速度列表。

    返回:
    - list[Chromesome]: 初始化后的种群（染色体列表）。
    """
    population: list[Chromesome] = []  # 创建一个空的种群列表
    for _ in range(population_size):  # 遍历种群大小
        # 创建新的染色体并添加到种群中
        population.append(Chromesome(target_num=target_num, target_type_num=target_type_num, US_list=US_list, UA_list=UA_list, 
                                     agent_positions=agent_positions, target_positions=target_positions, agent_velocities=agent_velocities))
    return population


def crossover_chromesome(father: Chromesome, mother: Chromesome) -> list[Chromesome]:
    """
    对两个染色体进行交叉操作，生成新的染色体。

    参数:
    - father (Chromesome): 父染色体。
    - mother (Chromesome): 母染色体。

    返回:
    - list[Chromesome]: 由父母染色体生成的两个子染色体。
    """
    child1 = father.copy()  # 复制父染色体
    child2 = mother.copy()  # 复制母染色体
    child1.sort_genes_by_target()  # 按目标对child1的基因排序
    child2.sort_genes_by_target()  # 按目标对child2的基因排序

    gene_num = len(child1)  # 获取基因数量
    # 随机选择两个交叉点
    [crossover_sites1, crossover_sites2] = np.random.choice(np.arange(gene_num + 1), 2, replace=False)
    if crossover_sites1 > crossover_sites2:  # 确保crossover_sites1小于crossover_sites2
        tmp = crossover_sites1
        crossover_sites1 = crossover_sites2
        crossover_sites2 = tmp
    tmp_agent_id_list = [gene.agent_id_ for gene in child1.genes_]  # 临时保存child1的agent_id列表
    for index in range(crossover_sites1, crossover_sites2):  # 在交叉点范围内交换基因的agent_id
        child1[index].agent_id_ = child2[index].agent_id_
        child2[index].agent_id_ = tmp_agent_id_list[index]

    child1.sort_genes_by_order()  # 按顺序对child1的基因排序
    child2.sort_genes_by_order()  # 按顺序对child2的基因排序

    return [child1, child2]  # 返回生成的两个子染色体


def mutate_chromesome_order_of_tasks(parent: Chromesome) -> Chromesome:
    # 复制父染色体，生成子染色体
    child = parent.copy()
    
    # 按目标排序基因
    child.sort_genes_by_target()
    
    # 生成目标索引列表
    target_index_list = np.arange(child.target_num_)
    
    # 生成目标类型列表
    target_type_list = np.arange(child.target_type_num_)
    
    # 获取目标类型数量
    target_type_num = child.target_type_num_
    
    # 打乱目标索引列表的顺序
    np.random.shuffle(target_index_list)
    
    # 获取当前基因的顺序列表
    tmp_order_list = [gene.order_ for gene in child.genes_]
    
    # 按照打乱后的目标索引列表重新设置基因的顺序
    for index, target_index in enumerate(target_index_list):
        for offset in target_type_list:
            child[index].order_ = tmp_order_list[target_index * target_type_num + offset]
    
    # 按顺序对基因进行排序
    child.sort_genes_by_order()
    
    # 返回变异后的子染色体
    return child


def mutate_chromesome_task(parent: Chromesome) -> Chromesome:
    # 复制父染色体，生成子染色体
    child = parent.copy()
    
    # 按任务排序基因
    child.sort_genes_by_task()
    
    # 获取目标数量
    target_num = child.target_num_
    
    # 随机选择一个突变位置
    mutation_site = np.random.choice(child.target_type_num_)
    
    # 在目标内生成智能体（机器人）索引列表
    in_target_agent_index_list = np.arange(target_num)
    
    # 打乱智能体（机器人）索引列表的顺序
    np.random.shuffle(in_target_agent_index_list)
    
    # 获取当前突变位点的智能体（机器人）列表
    tmp_agent_list = [child.genes_[mutation_site * target_num + i].agent_id_ for i in range(target_num)]
    
    # 按照打乱后的智能体（机器人）索引列表重新设置智能体（机器人）ID
    for index, agent_index in enumerate(in_target_agent_index_list):
        child[mutation_site * target_num + index].agent_id_ = tmp_agent_list[agent_index]
    
    # 按顺序对基因进行排序
    child.sort_genes_by_order()
    
    # 返回变异后的子染色体
    return child


def mutate_single_gene_agent(parent: Chromesome) -> Chromesome:
    # 创建一个父代染色体的副本
    child = parent.copy()
    # 随机选择一个基因进行突变
    mutation_site = np.random.choice(len(child))
    # 根据任务类型选择相应的智能体（机器人）列表
    if child[mutation_site].task_type_ == 1:
        tmp_agent_list = [agent_id for agent_id in child.UA_list_]
    else:
        tmp_agent_list = [agent_id for agent_id in child.US_list_]
    # 从智能体（机器人）列表中移除当前智能体（机器人）ID
    tmp_agent_list.remove(child[mutation_site].agent_id_)
    # 随机选择一个新的智能体（机器人）ID进行替换
    child[mutation_site].agent_id_ = np.random.choice(tmp_agent_list)
    return child


def crossover(parents: list[Chromesome], children_num: int) -> list[Chromesome]:
    # 随机选择指定数量的父代染色体
    selected_parent_index = np.random.choice(np.arange(len(parents)), children_num, replace=False)
    parents = [parents[i] for i in selected_parent_index]
    parents_len = len(parents)
    # 检查父代数量是否为奇数
    parity = parents_len % 2 == 1
    children: list[Chromesome] = []
    # 进行交叉操作，生成子代染色体
    for i in range(0, parents_len - 1, 2):
        children.extend(crossover_chromesome(parents[i], parents[i + 1]))
    # 如果父代数量为奇数，保留最后一个父代染色体的副本
    if parity:
        children.append(parents[len(parents) - 1].copy())
    return children


def mutate(parents: list[Chromesome], children_num: int) -> list[Chromesome]:
    # 随机选择一个父代染色体
    parent = parents[np.random.choice(len(parents))]
    children: list[Chromesome] = []
    for _ in range(children_num):
        # 随机选择一种变异类型
        mutation_type = np.random.choice(3)
        if mutation_type == 0:
            # 变异单个基因
            children.append(mutate_single_gene_agent(parent))
        elif mutation_type == 1:
            # 变异染色体任务
            children.append(mutate_chromesome_task(parent))
        elif mutation_type == 2:
            # 变异染色体任务顺序
            children.append(mutate_chromesome_order_of_tasks(parent))
    return children


def select(population: list[Chromesome]) -> list[Chromesome]:
    # 使用双指针法优化时间复杂度为 nlogn
    # 根据最小化目标，对每一代的染色体按适应度值线性分配在0到1之间
    child_population: list[Chromesome] = []
    # 根据适应度值对染色体进行排序，适应度值高的排在前面
    sorted_indices = sorted(range(len(population)), key=lambda index : population[index].fitness(), reverse=True)
    # 计算每个染色体的步长
    step = 1 / len(population)
    # 生成适应度值数组
    fitness = np.arange(step, 1 + step, step)
    # 计算适应度值的概率
    fitness_probability = fitness / sum(fitness)
    # 计算适应度值的累积分布函数
    fitness_probability = np.cumsum(fitness_probability)
    # 生成随机概率数组，并排序
    random_probability = np.sort(np.random.rand(len(population)))
    fitness_index = 0
    random_index = 0
    # 使用双指针法选择子代
    while fitness_index < len(fitness_probability) and random_index < len(random_probability):
        if random_probability[random_index] < fitness_probability[fitness_index]:
            # 选择适应度值高的染色体作为子代
            child_population.append(population[sorted_indices[fitness_index]].copy())
            random_index += 1
        else:
            fitness_index += 1
    # 清空原始种群
    del population[:]
    return child_population


def elite_select(population: list[Chromesome], elite_num: int) -> list[Chromesome]:
    # 使用 heapq.nsmallest 函数从种群中选出适应度最高的前 elite_num 个个体
    # 这里 lambda 表达式 individual : individual.fitness() 用于获取个体的适应度值
    elites = heapq.nsmallest(elite_num, population, lambda individual : individual.fitness())
    
    # 对选出的精英个体进行复制，以确保返回的精英个体是原个体的副本而不是引用
    return [elite.copy() for elite in elites]


def adaptive_evolve(target_num: int, target_type_num: int, population_size: int, elite_num: int, iteration_num: int, 
                    US_list: list[int], UA_list: list[int], agent_positions: list[Position],target_positions: list[Position], 
                    agent_velocities: list[float]) -> tuple[Chromesome, list[float], list[float]]:
    """
    自适应进化算法的主函数。

    参数：
    - target_num: 目标数量
    - target_type_num: 目标类型数量
    - population_size: 种群大小
    - elite_num: 精英个体数量
    - iteration_num: 迭代次数
    - US_list: US列表
    - UA_list: UA列表
    - agent_positions: 智能体（机器人）位置列表
    - target_positions: 目标位置列表
    - agent_velocities: 智能体（机器人）速度列表

    返回值：
    - 最优解（Chromesome类型）
    - 平均适应度列表
    - 最佳适应度列表
    """
    
    # 初始化种群
    population = population_initialization(target_num, target_type_num, population_size, 
                                           US_list, UA_list, agent_positions, target_positions, agent_velocities)
    avg_fitness_list = []  # 用于存储每次迭代的平均适应度
    best_fitness_list = []  # 用于存储每次迭代的最佳适应度
    
    for iteration in range(iteration_num):
        # 根据当前迭代次数动态调整交叉和变异的个体数量
        crossover_num = round((population_size - elite_num) * math.exp(-iteration / iteration_num))
        mutation_num = population_size - elite_num - crossover_num

        # 精英选择，保留适应度最高的个体
        elite_parents = elite_select(population, elite_num)
        # 选择操作，选择出适应度较高的个体进行交叉和变异
        population = select(population)
        # 交叉操作，生成新的个体
        crossover_offspring = crossover(population, crossover_num)
        # 变异操作，生成新的个体
        mutation_offspring = mutate(population, mutation_num)
        
        # 清空原种群并添加新的个体
        del population[:]
        population.extend(elite_parents)
        population.extend(crossover_offspring)
        population.extend(mutation_offspring)

        # 计算当前种群的平均适应度和最佳适应度
        avg_fitness_list.append(sum([individual.fitness() for individual in population]) / population_size)
        best_fitness_list.append(min([individual.fitness() for individual in population]))
        
        # 垃圾回收，释放内存
        gc.collect()
        print(f"iteration {iteration}")
    
    # 找到适应度最好的个体作为最优解
    best_solution = min(population, key=lambda individual : individual.fitness())
    return best_solution, avg_fitness_list, best_fitness_list


def draw(solution: Chromesome, agent_positions: list[Position], target_positions: list[Position], agent_velocities: list[float], 
         avg_fitness_list: list[float], best_fitness_list: list[float]) -> None:
    # x 轴的数据点，表示每一代的编号
    x = range(len(avg_fitness_list))
    
    # 第一个子图：平均适应度
    plt.subplot(2, 2, 1)
    plt.title("average fitness")
    plt.plot(x, avg_fitness_list)
    
    # 第二个子图：最佳适应度
    plt.subplot(2, 2, 3)
    plt.title("best fitness")
    plt.plot(x, best_fitness_list)
    
    # 第三个子图：任务分配情况
    plt.subplot(1, 2, 2)
    plt.title("task allocation")
    
    # 为每个智能体（机器人）生成一个随机颜色
    colors: list[tuple[float, float, float]] = [tuple([np.random.rand() for _ in range(3)]) for _ in range(len(agent_positions))]
    
    # 提取目标位置的 x 和 y 坐标
    target_x = []
    target_y = []
    for position in target_positions:
        target_x.append(position.x_)
        target_y.append(position.y_)
    
    # 绘制目标位置，用黑色叉标记
    plt.scatter(target_x, target_y, c="black", marker='x')
    
    # 绘制每个智能体（机器人）的位置，用各自的颜色标记
    for i, position in enumerate(agent_positions):
        plt.scatter(position.x_, position.y_, c=colors[i])
    
    # 绘制每个智能体（机器人）的任务路径
    for genes in solution.get_gene_set_by_agent():
        agent_id = genes[0].agent_id_
        
        # 初始化路径的 x 和 y 坐标，起点是智能体（机器人）的位置
        trajectory_x = [agent_positions[agent_id].x_]
        trajectory_y = [agent_positions[agent_id].y_]
        
        # 添加路径上的每个目标位置
        trajectory_x.extend([target_positions[gene.target_id_].x_ for gene in genes])
        trajectory_y.extend([target_positions[gene.target_id_].y_ for gene in genes])
        
        # 计算路径总距离
        distance = 0
        for j in range(1, len(trajectory_x)):
            distance += math.sqrt((trajectory_x[j] - trajectory_x[j - 1]) ** 2 + (trajectory_y[j] - trajectory_y[j - 1]) ** 2)
        
        # 计算完成任务所需时间
        time = distance / agent_velocities[agent_id]
        
        # 绘制路径，并在图例中标注智能体（机器人）信息（ID 和速度），路径信息（总距离和时间）
        plt.plot(trajectory_x, trajectory_y, label=f'Agent <{agent_id}, {agent_velocities[agent_id]}>: <{distance}, {time}>', c=colors[agent_id])
    
    # 显示所有图
    plt.show()


def draw_target_allocation(solution: Chromesome, agent_positions: list[Position], target_positions: list[Position], agent_velocities: list[float], 
         avg_fitness_list: list[float], best_fitness_list: list[float]) -> None:
    # 计算大于等于num的最小平方数
    def min_sqrt_num(num: int) -> int:
        res = 1
        while res * res < num:
            res += 1
        return res

    # 线性插值计算
    def interpolation(x1: float, y1: float, x2: float, y2: float, s: float) -> tuple[float, float]:
        return (x1 + (x2 - x1) * s, y1 + (y2 - y1) * s)

    # 确定绘图的行数
    agent_plt_rows = min_sqrt_num(len(agent_positions) + 1)
    x = range(len(avg_fitness_list))

    # 绘制平均适应度图
    plt.subplot(2, agent_plt_rows + 1, 1)
    plt.title("average fitness")
    plt.plot(x, avg_fitness_list)

    # 绘制最佳适应度图
    plt.subplot(2, agent_plt_rows + 1, agent_plt_rows + 2)
    plt.title("best fitness")
    plt.plot(x, best_fitness_list)

    # 为每个agent生成一个随机颜色
    colors: list[tuple[float, float, float]] = [tuple([np.random.rand() for _ in range(3)]) for _ in range(len(agent_positions))]
    target_x = []
    target_y = []

    # 提取目标位置的x和y坐标
    for position in target_positions:
        target_x.append(position.x_)
        target_y.append(position.y_)

    # 为每个agent绘制轨迹
    for genes in solution.get_gene_set_by_agent():
        agent_id = genes[0].agent_id_
        trajectory_x = [agent_positions[agent_id].x_]
        trajectory_y = [agent_positions[agent_id].y_]
        trajectory_x.extend([target_positions[gene.target_id_].x_ for gene in genes])
        trajectory_y.extend([target_positions[gene.target_id_].y_ for gene in genes])

        # 计算轨迹的总距离
        distance = 0
        for j in range(1, len(trajectory_x)):
            distance += math.sqrt((trajectory_x[j] - trajectory_x[j - 1]) ** 2 + (trajectory_y[j] - trajectory_y[j - 1]) ** 2)

        # 计算总时间
        time = distance / agent_velocities[agent_id]

        # 为每个agent绘制单独的子图
        plt.subplot(agent_plt_rows, agent_plt_rows + 1, agent_id // agent_plt_rows * (agent_plt_rows + 1) + 2 + agent_id % agent_plt_rows)
        plt.title(f"Agent {agent_id}")
        
        # 绘制带箭头的轨迹
        j = 1
        for i in range(len(trajectory_x)):
            if i == 0 or (trajectory_x[i - 1] == trajectory_x[i] and trajectory_y[i - 1] == trajectory_y[i]):
                continue
            plt.annotate("", xy=(trajectory_x[i], trajectory_y[i]), xytext=(trajectory_x[i - 1], trajectory_y[i - 1]), arrowprops=dict(arrowstyle='->', color=colors[agent_id]))
            interpolation_x, interpolation_y = interpolation(trajectory_x[i - 1], trajectory_y[i - 1], trajectory_x[i], trajectory_y[i], 0.333)
            plt.text(interpolation_x, interpolation_y, s=f'{j}')
            j += 1
        plt.scatter(target_x, target_y, c="black", marker='x')
        plt.scatter(trajectory_x[0], trajectory_y[0], c=colors[agent_id])

    # 绘制所有目标位置
    plt.subplot(agent_plt_rows, agent_plt_rows + 1, agent_plt_rows ** 2 + agent_plt_rows)
    for i, position in enumerate(target_positions):
        plt.scatter(position.x_, position.y_, c='black', marker='x')
        plt.text(position.x_ + 4, position.y_ + 4, s=f'{i}')
    plt.show()


def test():
    # 目标数量
    target_num = 15
    # 种群大小
    population_size = 100
    # 精英数量
    elite_num = 4
    # 迭代次数
    iteration_num = 300
    # 区域范围
    area_length = 10000

    # 定义智能体（机器人）的位置列表
    agent_positions: list[Position] = []
    # 定义目标位置列表
    target_positions: list[Position] = []
    # 定义智能体（机器人）的速度列表
    agent_velocities: list[float] = []

    # 初始化智能体（机器人）的位置
    agent_positions.append(Position(0, 0))  # agent_id: 0
    agent_positions.append(Position(0, area_length))
    agent_positions.append(Position(area_length, area_length))
    agent_positions.append(Position(area_length, 0))
    agent_positions.append(Position(0, area_length / 2))
    agent_positions.append(Position(area_length / 2, area_length))
    agent_positions.append(Position(area_length, area_length / 2))
    agent_positions.append(Position(area_length / 2, 0))  # agent_id: 7

    # 初始化智能体（机器人）的速度
    agent_velocities.append(70)
    agent_velocities.append(80)
    agent_velocities.append(90)
    agent_velocities.append(100)
    agent_velocities.append(60)
    agent_velocities.append(70)
    agent_velocities.append(80)
    agent_velocities.append(90)

    # 随机生成目标位置
    for _ in range(target_num):
        target_positions.append(Position(area_length, area_length, True))

    # 调用自适应进化算法进行求解
    solution, avg_fitness_list, best_fitness_list = adaptive_evolve(
        target_num,
        3,
        population_size,
        elite_num,
        iteration_num,
        [0, 1, 2, 3],  # 前四个智能体的ID
        [4, 5, 6, 7],  # 后四个智能体的ID
        agent_positions,
        target_positions,
        agent_velocities
    )

    # 输出最终解和其适应度值
    print(solution)
    print(solution.fitness())

    # 绘制目标分配图
    draw_target_allocation(solution, agent_positions, target_positions, agent_velocities, avg_fitness_list, best_fitness_list)
    # 绘制其他相关图表
    draw(solution, agent_positions, target_positions, agent_velocities, avg_fitness_list, best_fitness_list)


def test_example():
    # 目标数量
    target_num = 2
    # 种群大小
    population_size = 100
    # 精英数量
    elite_num = 4
    # 迭代次数
    iteration_num = 300
    
    # 智能体（机器人）的位置列表
    agent_positions: list[Position] = []
    # 目标的位置列表
    target_positions: list[Position] = []
    # 智能体（机器人）的速度列表
    agent_velocities: list[float] = []
    
    # 添加智能体（机器人）的位置 (x=2500, y=0)，智能体（机器人）ID: 0
    agent_positions.append(Position(2500, 0))
    # 再次添加相同的位置 (x=2500, y=0)
    agent_positions.append(Position(2500, 0))
    # 再次添加相同的位置 (x=2500, y=0)
    agent_positions.append(Position(2500, 0))
    
    # 添加智能体（机器人）的速度
    agent_velocities.append(70)
    # 添加另一个智能体（机器人）的速度
    agent_velocities.append(80)
    # 添加第三个智能体（机器人）的速度
    agent_velocities.append(70)
    
    # US类型智能体（机器人）列表
    US_list = [0, 1]
    # UA类型智能体（机器人）列表
    UA_list = [1, 2]
    
    # 添加目标的位置 (x=1000, y=3400)
    target_positions.append(Position(1000, 3400))
    # 添加另一个目标的位置 (x=4500, y=4000)
    target_positions.append(Position(4500, 4000))
    
    # 调用自适应进化函数，得到解决方案，平均适应度列表和最佳适应度列表
    solution, avg_fitness_list, best_fitness_list = adaptive_evolve(
        target_num, 
        3,  # 智能体（机器人）数量
        population_size, 
        elite_num, 
        iteration_num,
        US_list, 
        UA_list, 
        agent_positions, 
        target_positions, 
        agent_velocities
    )
    
    # 打印解决方案
    print(solution)
    # 打印解决方案的适应度
    print(solution.fitness())
    
    # 绘制结果，包括解决方案，智能体（机器人）位置，目标位置，智能体（机器人）速度，平均适应度列表和最佳适应度列表
    draw(solution, agent_positions, target_positions, agent_velocities, avg_fitness_list, best_fitness_list)


#test_example()
test()
