import numpy as np
import functools
import math
import heapq
import matplotlib.pyplot as plt
import gc


class Position:
    def __init__(self, x: float, y: float, random: bool=False) -> None:
        # 初始化位置对象，x 和 y 是坐标，如果 random 为 True，则 x 和 y 会乘以一个随机数
        self.x_ = x
        self.y_ = y
        if random:
            self.x_ *= np.random.rand()
            self.y_ *= np.random.rand()
    
    def distance_to(self, pos) -> float:
        # 计算当前对象与另一个位置对象 pos 之间的欧几里得距离
        return math.sqrt((self.x_ - pos.x_) ** 2 + (self.y_ - pos.y_) ** 2)


class Gene:
    
    def __init__(self, order: int, target_id: int, task_type: int, agent_id: int) -> None:
        # 初始化基因对象，包含序号、目标 ID、任务类型和智能体（机器人） ID
        self.order_ = order
        self.target_id_ = target_id
        self.task_type_ = task_type
        self.agent_id_ = agent_id

    def copy(self):
        # 返回当前基因对象的一个副本
        return Gene(self.order_, self.target_id_, self.task_type_, self.agent_id_)


def sort_key_by_order(gene: Gene):
    # 返回基因对象的序号，用于排序
    return gene.order_


def sort_cmp_by_target(gene1: Gene, gene2: Gene):
    # 比较两个基因对象的目标 ID 和任务类型，用于排序
    if gene1.target_id_ < gene2.target_id_:
        return -1
    if gene1.target_id_ == gene2.target_id_ and gene1.task_type_ < gene2.task_type_:
        return -1
    return 1


def sort_cmp_by_agent(gene1: Gene, gene2: Gene):
    # 比较两个基因对象的智能体（机器人） ID 和序号，用于排序
    if gene1.agent_id_ < gene2.agent_id_:
        return -1
    if gene1.agent_id_ == gene2.agent_id_ and gene1.order_ < gene2.order_:
        return -1
    return 1


def sort_cmp_by_task(gene1: Gene, gene2: Gene):
    # 比较两个基因对象的任务类型和序号，用于排序
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


class Agent:
    def __init__(self, target_num: int, target_type_num: int, population_size: int, US_list: list[int],
                UA_list: list[int], agent_positions: list[Position], target_positions: list[Position],
                agent_velocities: list[float], total_iteration: int, elite_num: int):
        """
        初始化Agent类的实例。

        参数:
        target_num (int): 目标数量。
        target_type_num (int): 目标类型数量。
        population_size (int): 种群大小。
        US_list (list[int]): US类型智能体（机器人）列表。
        UA_list (list[int]): UA类型智能体（机器人）列表。
        agent_positions (list[Position]): 智能体（机器人）的位置列表。
        target_positions (list[Position]): 目标的位置列表。
        agent_velocities (list[float]): 智能体（机器人）的速度列表。
        total_iteration (int): 总迭代次数。
        elite_num (int): 精英数量。
        """
        # population_size: 种群大小，即个体数量
        self.population_size = population_size
        
        # local_population: 本地种群列表，用于存储Chromesome对象
        self.local_population: list[Chromesome] = []
        
        # 使用给定的参数初始化种群中的每个个体（Chromesome对象）
        for _ in range(population_size):
            self.local_population.append(Chromesome(target_num=target_num, target_type_num=target_type_num, US_list=US_list, UA_list=UA_list, 
                                        agent_positions=agent_positions, target_positions=target_positions, agent_velocities=agent_velocities))
        
        # current_iteration: 当前迭代次数，初始化为0
        self.current_iteration = 0
        
        # total_iteration: 总迭代次数
        self.total_iteration = total_iteration
        
        # elite_num: 精英个体数量，用于选择最优个体
        self.elite_num = elite_num

    def crossover_chromesome(self, father: Chromesome, mother: Chromesome) -> list[Chromesome]:
        # 创建两个子代染色体，分别是父代和母代的副本
        child1 = father.copy()
        child2 = mother.copy()

        # 根据目标对基因进行排序
        child1.sort_genes_by_target()
        child2.sort_genes_by_target()

        # 获取基因数量
        gene_num = len(child1)

        # 随机选择两个交叉点
        [crossover_sites1, crossover_sites2] = np.random.choice(np.arange(gene_num + 1), 2, replace=False)
        
        # 确保第一个交叉点小于第二个交叉点，如果不是则交换它们
        if crossover_sites1 > crossover_sites2:
            tmp = crossover_sites1
            crossover_sites1 = crossover_sites2
            crossover_sites2 = tmp

        # 保存child1的基因的智能体（机器人）ID列表
        tmp_agent_id_list = [gene.agent_id_ for gene in child1.genes_]

        # 在交叉点之间交换父代和母代的基因
        for index in range(crossover_sites1, crossover_sites2):
            child1[index].agent_id_ = child2[index].agent_id_
            child2[index].agent_id_ = tmp_agent_id_list[index]

        # 根据顺序对基因重新排序
        child1.sort_genes_by_order()
        child2.sort_genes_by_order()

        # 返回两个子代染色体
        return [child1, child2]

    def mutate_chromesome_order_of_tasks(self, parent: Chromesome) -> Chromesome:
        # 创建父代染色体的副本作为子代染色体
        child = parent.copy()
        
        # 根据目标对基因进行排序
        child.sort_genes_by_target()
        
        # 创建目标索引列表，从0到目标数减1
        target_index_list = np.arange(child.target_num_)
        
        # 创建目标类型列表，从0到目标类型数减1
        target_type_list = np.arange(child.target_type_num_)
        
        # 获取目标类型的数量
        target_type_num = child.target_type_num_
        
        # 随机打乱目标索引列表
        np.random.shuffle(target_index_list)
        
        # 获取子代染色体的基因的顺序列表
        tmp_order_list = [gene.order_ for gene in child.genes_]
        
        # 遍历打乱后的目标索引列表
        for index, target_index in enumerate(target_index_list):
            # 遍历目标类型列表
            for offset in target_type_list:
                # 更新子代染色体的基因顺序
                child[index].order_ = tmp_order_list[target_index * target_type_num + offset]
        
        # 根据更新后的顺序对基因进行排序
        child.sort_genes_by_order()
        
        # 返回变异后的子代染色体
        return child

    def mutate_chromesome_task(self, parent: Chromesome) -> Chromesome:
        # 复制父染色体，生成一个新的子染色体
        child = parent.copy()
        
        # 按任务排序子染色体的基因
        child.sort_genes_by_task()
        
        # 获取目标数量
        target_num = child.target_num_
        
        # 随机选择一个突变位置的索引
        mutation_site = np.random.choice(child.target_type_num_)
        
        # 生成一个从0到target_num-1的整数数组，并打乱顺序
        in_target_agent_index_list = np.arange(target_num)
        np.random.shuffle(in_target_agent_index_list)
        
        # 获取当前突变位置上每个基因的agent_id_
        tmp_agent_list = [child.genes_[mutation_site * target_num + i].agent_id_ for i in range(target_num)]
        
        # 对于打乱顺序后的每个agent_index，将对应位置的agent_id_替换为tmp_agent_list中随机选取的agent_id_
        for index, agent_index in enumerate(in_target_agent_index_list):
            child[mutation_site * target_num + index].agent_id_ = tmp_agent_list[agent_index]
        
        # 根据顺序重新排序子染色体的基因
        child.sort_genes_by_order()
        
        # 返回突变后的子染色体
        return child

    def mutate_single_gene_agent(self, parent: Chromesome) -> Chromesome:
        # 复制父染色体，生成一个新的子染色体
        child = parent.copy()
        
        # 从子染色体中随机选择一个突变位置的索引
        mutation_site = np.random.choice(len(child))
        
        # 根据所选择基因的任务类型决定使用的智能体（机器人）列表
        if child[mutation_site].task_type_ == 1:
            # 如果任务类型为1，则使用UA_list_作为智能体（机器人）列表
            tmp_agent_list = [agent_id for agent_id in child.UA_list_]
        else:
            # 如果任务类型不是1，则使用US_list_作为智能体（机器人）列表
            tmp_agent_list = [agent_id for agent_id in child.US_list_]
        
        # 从智能体（机器人）列表中移除当前基因的agent_id_，以确保不会重复
        tmp_agent_list.remove(child[mutation_site].agent_id_)
        
        # 随机选择一个新的agent_id_，并赋值给选定突变位置的基因
        child[mutation_site].agent_id_ = np.random.choice(tmp_agent_list)
        
        # 返回突变后的子染色体
        return child

    def crossover(self, children_num: int) -> list[Chromesome]:
        # 从当前种群中随机选择指定数量的父染色体索引
        selected_parent_index = np.random.choice(np.arange(self.population_size), children_num, replace=False)
        
        # 根据选择的索引获取父染色体
        parents = [self.local_population[i] for i in selected_parent_index]
        
        # 计算父染色体的数量
        parents_len = len(parents)
        
        # 判断父染色体的数量是否为奇数
        parity = parents_len % 2 == 1
        
        # 初始化一个空列表，用于存储生成的子染色体
        children: list[Chromesome] = []
        
        # 对父染色体进行配对并生成子染色体
        for i in range(0, parents_len - 1, 2):
            # 对每对父染色体进行交叉操作，生成两个子染色体，并将它们添加到子染色体列表中
            children.extend(self.crossover_chromesome(parents[i], parents[i + 1]))
        
        # 如果父染色体数量为奇数，单独处理最后一个父染色体
        if parity:
            # 将最后一个父染色体复制一份，添加到子染色体列表中
            children.append(parents[len(parents) - 1].copy())
        
        # 返回生成的子染色体列表
        return children

    def mutate(self, children_num: int) -> list[Chromesome]:
        # 从当前种群中随机选择一个父染色体
        parent = self.local_population[np.random.choice(self.population_size)]
        
        # 初始化一个空列表，用于存储生成的子染色体
        children: list[Chromesome] = []
        
        # 生成指定数量的子染色体
        for _ in range(children_num):
            # 随机选择一种突变类型（0, 1, 或 2）
            mutation_type = np.random.choice(3)
            
            # 根据选择的突变类型生成对应的子染色体
            if mutation_type == 0:
                # 使用单基因智能体（机器人）突变方法
                children.append(self.mutate_single_gene_agent(parent))
            elif mutation_type == 1:
                # 使用任务突变方法
                children.append(self.mutate_chromesome_task(parent))
            elif mutation_type == 2:
                # 使用任务顺序突变方法
                children.append(self.mutate_chromesome_order_of_tasks(parent))
        
        # 返回生成的子染色体列表
        return children

    def select(self):
        # 双指针方法用于优化选择操作的时间复杂度为 O(n log n)
        # 根据最小化目标，将每代染色体的适应度值线性地分配在0到1之间
        child_population: list[Chromesome] = []  # 初始化一个空列表，用于存储选择后的子染色体

        # 获取当前种群的所有染色体索引，并根据染色体的适应度进行排序
        # 排序规则是适应度高的在前（适应度是最大化目标的情况下）
        sorted_indices = sorted(range(self.population_size), key=lambda index: self.local_population[index].fitness(), reverse=True)
        
        # 计算每个染色体的适应度权重
        step = 1 / self.population_size  # 每个染色体的适应度步长
        fitness = np.arange(step, 1 + step, step)  # 生成线性分布的适应度值（从step到1之间）
        
        # 将适应度值归一化为概率
        fitness_probability = fitness / sum(fitness)  # 计算适应度的概率
        fitness_probability = np.cumsum(fitness_probability)  # 计算累计概率
        
        # 生成随机概率值，并对其进行排序
        random_probability = np.sort(np.random.rand(self.population_size))
        
        fitness_index = 0  # 指向适应度概率的索引
        random_index = 0  # 指向随机概率的索引
        
        # 使用双指针方法选择染色体
        while fitness_index < len(fitness_probability) and random_index < len(random_probability):
            if random_probability[random_index] < fitness_probability[fitness_index]:
                # 如果随机概率小于适应度概率，选择当前适应度的染色体
                child_population.append(self.local_population[sorted_indices[fitness_index]].copy())
                random_index += 1  # 移动到下一个随机概率
            else:
                # 否则，检查下一个适应度值
                fitness_index += 1  # 移动到下一个适应度概率
        
        # 更新种群为选择后的染色体
        del self.local_population[:]  # 清空当前种群
        self.local_population = child_population  # 将选择后的子染色体赋值给当前种群

    def elite_select(self, elite_num: int) -> list[Chromesome]:
        # 使用堆排序选择当前种群中适应度最好的若干个精英染色体
        # elite_num 指定了要选择的精英染色体的数量
        
        # 使用堆排序选择适应度最小的 elite_num 个染色体
        # heapq.nsmallest 返回适应度值最小的 elite_num 个染色体
        elites = heapq.nsmallest(elite_num, self.local_population, lambda individual: individual.fitness())
        
        # 复制选择出的精英染色体并返回
        return [elite.copy() for elite in elites]

    def adaptive_evolve(self, iteration_num: int) -> tuple[Chromesome, list[float], list[float]]:
        # 自适应演化方法，进行迭代演化以优化种群
        # 返回最优解、每代的平均适应度列表和每代的最佳适应度列表
        
        avg_fitness_list = []  # 初始化一个列表，用于存储每代的平均适应度
        best_fitness_list = []  # 初始化一个列表，用于存储每代的最佳适应度
        
        for iteration in range(iteration_num):
            # 计算当前迭代中进行交叉操作的染色体数量
            crossover_num = round((self.population_size - self.elite_num) * \
                                math.exp(-(iteration + self.current_iteration) / self.total_iteration))
            # 计算当前迭代中进行突变操作的染色体数量
            mutation_num = self.population_size - self.elite_num - crossover_num
            
            # 选择当前种群中的精英染色体
            elite_parents = self.elite_select(self.elite_num)
            
            # 选择当前种群的染色体（进行淘汰操作）
            self.select()
            
            # 执行交叉操作，生成交叉后代
            crossover_offspring = self.crossover(crossover_num)
            
            # 执行突变操作，生成突变后代
            mutation_offspring = self.mutate(mutation_num)
            
            # 清空当前种群，并将精英染色体、交叉后代和突变后代添加到种群中
            del self.local_population[:]  # 清空当前种群
            self.local_population.extend(elite_parents)  # 添加精英染色体
            self.local_population.extend(crossover_offspring)  # 添加交叉后代
            self.local_population.extend(mutation_offspring)  # 添加突变后代
            
            # 计算当前种群的平均适应度，并将其添加到平均适应度列表中
            avg_fitness_list.append(sum([individual.fitness() for individual in self.local_population]) / self.population_size)
            
            # 计算当前种群的最佳适应度，并将其添加到最佳适应度列表中
            best_fitness_list.append(min([individual.fitness() for individual in self.local_population]))
            
            # 进行垃圾回收，释放不再使用的内存
            gc.collect()
            
            # 打印当前迭代的编号
            print(f"iteration {iteration}")
        
        # 更新当前迭代次数
        self.current_iteration += iteration_num
        
        # 选择当前种群中的最佳解
        best_solution = min(self.local_population, key=lambda individual: individual.fitness())
        
        # 返回最佳解、每代的平均适应度列表和每代的最佳适应度列表
        return best_solution, avg_fitness_list, best_fitness_list

    def select_topk(self, k: int) -> list[Chromesome]:
        # 选择当前种群中适应度最好的前 k 个染色体
        # 如果 k 大于当前种群的规模，则将 k 调整为种群规模
        if k > self.population_size:
            k = self.population_size
        
        # 使用堆排序选择适应度最小的 k 个染色体
        # heapq.nsmallest 返回适应度最小的 k 个染色体
        topk = heapq.nsmallest(k, self.local_population, lambda individual: individual.fitness())
        
        # 复制选择出的染色体，并返回
        return [individual.copy() for individual in topk]


    def update_local_population(self, remote_population: list[Chromesome]):
        # 更新当前种群，将远程种群中的染色体加入到本地种群中
        # 然后从本地种群中选择适应度最好的染色体，保持种群规模
        
        # 将远程种群中的每个染色体复制后加入到本地种群中
        self.local_population.extend([individual.copy() for individual in remote_population])
        
        # 使用堆排序从更新后的种群中选择适应度最小的染色体
        # 保持本地种群的规模为 self.population_size
        self.local_population = heapq.nsmallest(self.population_size, self.local_population, lambda individual: individual.fitness())

def adaptive_co_evolve(target_num: int, target_type_num: int, population_size: int, elite_num: int, total_iteration: int,
                    iteration_unit: int, US_list: list[int], UA_list: list[int], agent_positions: list[Position],
                    target_positions: list[Position], agent_velocities: list[float], share_num: int) -> \
                    tuple[Chromesome, list[list[float]], list[list[float]]]:
    # 自适应协同进化方法
    # 参数:
    # target_num: 目标数量
    # target_type_num: 目标类型数量
    # population_size: 种群规模
    # elite_num: 精英个体数量
    # total_iteration: 总迭代次数
    # iteration_unit: 每个进化单位的迭代次数
    # US_list: US类型智能体（机器人）列表
    # UA_list: UA类型智能体（机器人）列表
    # agent_positions: 智能体（机器人）者的位置列表
    # target_positions: 目标的位置列表
    # agent_velocities: 智能体（机器人）者的速度列表
    # share_num: 每个智能体（机器人）者共享的染色体数量
    # 返回:
    # 最佳解列表、每个智能体（机器人）者的平均适应度列表、每个智能体（机器人）者的最佳适应度列表

    # 初始化智能体（机器人）者列表
    agents: list[Agent] = [
        Agent(target_num, 
            target_type_num, 
            population_size, 
            US_list, 
            UA_list, 
            agent_positions, 
            target_positions, 
            agent_velocities, 
            total_iteration, 
            elite_num) 
        for _ in range(len(agent_positions))
    ]
    
    # 初始化记录最佳解、平均适应度和最佳适应度的列表
    best_solutions: list[Chromesome] = []
    avg_fitnesses: list[list[float]] = [[] for _ in range(len(agents))]
    best_fitnesses: list[list[float]] = [[] for _ in range(len(agents))]
    share_pool: list[list[Chromesome]] = [[] for _ in range(len(agents))]

    # 进行总迭代次数 // 每个进化单位的迭代次数的循环
    for j in range(total_iteration // iteration_unit):
        print(f"==================================== iteration {j} ====================================")
        best_solutions.clear()  # 清空最佳解列表
        share_pool.clear()  # 清空共享池列表

        # 对每个智能体（机器人）者进行进化操作
        for i in range(len(agents)):
            print(f"--------------------------- agent {i} ---------------------------")
            
            # 进行自适应进化，获取每个智能体（机器人）者的最佳解、平均适应度和最佳适应度
            best_solution, avg_fitness, best_fitness = agents[i].adaptive_evolve(iteration_unit)
            
            # 将每个智能体（机器人）者的平均适应度和最佳适应度添加到对应列表中
            avg_fitnesses[i].extend(avg_fitness)
            best_fitnesses[i].extend(best_fitness)
            
            # 添加每个智能体（机器人）者的最佳解到最佳解列表中
            best_solutions.append(best_solution)

            # 将每个智能体（机器人）者选择的精英染色体添加到共享池中
            share_pool.append(agents[i].select_topk(share_num))
        
        # 更新每个智能体（机器人）者的本地种群，排除掉自己的共享池
        for i in range(len(agents)):
            # np.delete(share_pool, i, axis=0) 删除第 i 个智能体（机器人）者的共享池
            # flatten() 将二维数组展平成一维数组
            agents[i].update_local_population(np.delete(share_pool, i, axis=0).flatten())
    
    # 返回每个智能体（机器人）者的最佳解、平均适应度列表和最佳适应度列表
    return best_solutions, avg_fitnesses, best_fitnesses


def draw_fitness(avg_fitness_list: list[list[float]], best_fitness_list: list[list[float]]) -> None:
    # 绘制每个智能体（机器人）者的平均适应度和最佳适应度变化曲线
    # 参数:
    # avg_fitness_list: 每个智能体（机器人）者的平均适应度列表，每个子列表表示一个智能体（机器人）者的适应度随迭代次数变化的情况
    # best_fitness_list: 每个智能体（机器人）者的最佳适应度列表，每个子列表表示一个智能体（机器人）者的最佳适应度随迭代次数变化的情况
    
    rows = len(avg_fitness_list)  # 获取智能体（机器人）者的数量，即需要绘制的行数
    x = range(len(avg_fitness_list[0]))  # x 轴的刻度，表示迭代次数的范围
    
    # 遍历每个智能体（机器人）者，绘制其适应度图
    for i in range(rows):
        # 绘制平均适应度图
        plt.subplot(rows, 2, i * 2 + 1)  # 创建一个行数为 rows，列数为 2 的子图网格，并选择第 i*2+1 个子图位置
        plt.title(f"Agent{i + 1} average fitness")  # 设置子图的标题，表示智能体（机器人）者的平均适应度
        plt.plot(x, avg_fitness_list[i])  # 绘制平均适应度曲线
    
        # 绘制最佳适应度图
        plt.subplot(rows, 2, i * 2 + 2)  # 选择第 i*2+2 个子图位置
        plt.title(f"Agent{i + 1} best fitness")  # 设置子图的标题，表示智能体（机器人）者的最佳适应度
        plt.plot(x, best_fitness_list[i])  # 绘制最佳适应度曲线
    
    plt.show()  # 显示绘制的所有子图


def draw_fitness(avg_fitness_list: list[list[float]], best_fitness_list: list[list[float]]) -> None:
    # 绘制每个智能体（机器人）者的平均适应度和最佳适应度变化曲线
    # 参数:
    # avg_fitness_list: 每个智能体（机器人）者的平均适应度列表。每个子列表包含一个智能体（机器人）者在不同迭代次数下的平均适应度。
    # best_fitness_list: 每个智能体（机器人）者的最佳适应度列表。每个子列表包含一个智能体（机器人）者在不同迭代次数下的最佳适应度。

    rows = len(avg_fitness_list)  # 获取智能体（机器人）者的数量，这将决定子图的行数
    x = range(len(avg_fitness_list[0]))  # x 轴刻度，表示迭代次数的范围
    
    # 遍历每个智能体（机器人）者并绘制其适应度曲线
    for i in range(rows):
        # 绘制当前智能体（机器人）者的平均适应度
        plt.subplot(rows, 2, i * 2 + 1)  # 创建一个 (rows x 2) 的子图网格，选择第 i*2+1 个位置
        plt.title(f"Agent{i + 1} average fitness")  # 设置子图标题，表示智能体（机器人）者的平均适应度
        plt.plot(x, avg_fitness_list[i])  # 绘制当前智能体（机器人）者的平均适应度曲线
        
        # 绘制当前智能体（机器人）者的最佳适应度
        plt.subplot(rows, 2, i * 2 + 2)  # 选择第 i*2+2 个位置
        plt.title(f"Agent{i + 1} best fitness")  # 设置子图标题，表示智能体（机器人）者的最佳适应度
        plt.plot(x, best_fitness_list[i])  # 绘制当前智能体（机器人）者的最佳适应度曲线

    plt.show()  # 显示所有绘制的子图


def draw_target_allocation(solution: Chromesome, agent_positions: list[Position], target_positions: list[Position], agent_velocities: list[float]) -> None:
    # 绘制目标分配图，包括每个智能体（机器人）者的轨迹和目标位置
    # 参数:
    # solution: 染色体解，包含每个智能体（机器人）者的目标分配情况
    # agent_positions: 智能体（机器人）者的位置列表，每个位置包含 x 和 y 坐标
    # target_positions: 目标的位置列表，每个位置包含 x 和 y 坐标
    # agent_velocities: 智能体（机器人）者的速度列表，表示每个智能体（机器人）者的速度

    def min_sqrt_num(num: int) -> int:
        # 计算大于或等于 num 的最小整数平方根值
        res = 1
        while res * res < num:
            res += 1
        return res

    def interpolation(x1: float, y1: float, x2: float, y2: float, s: float) -> tuple[float, float]:
        # 计算两个点之间的插值点
        return (x1 + (x2 - x1) * s, y1 + (y2 - y1) * s)

    # 计算绘图网格的行数，使其能容纳所有智能体（机器人）者的图
    agent_plt_rows = min_sqrt_num(len(agent_positions) + 1)

    # 为每个智能体（机器人）者生成随机颜色
    colors: list[tuple[float, float, float]] = [tuple([np.random.rand() for _ in range(3)]) for _ in range(len(agent_positions))]

    # 目标位置的 x 和 y 坐标
    target_x = [position.x_ for position in target_positions]
    target_y = [position.y_ for position in target_positions]

    # 遍历每个智能体（机器人）者的基因，绘制其轨迹
    for genes in solution.get_gene_set_by_agent():
        agent_id = genes[0].agent_id_  # 获取当前智能体（机器人）者的 ID
        trajectory_x = [agent_positions[agent_id].x_]  # 初始化轨迹的 x 坐标，包含智能体（机器人）者的起始位置
        trajectory_y = [agent_positions[agent_id].y_]  # 初始化轨迹的 y 坐标，包含智能体（机器人）者的起始位置
        
        # 添加目标位置到轨迹中
        trajectory_x.extend([target_positions[gene.target_id_].x_ for gene in genes])
        trajectory_y.extend([target_positions[gene.target_id_].y_ for gene in genes])
        
        # 计算轨迹的总距离
        distance = 0
        for j in range(1, len(trajectory_x)):
            distance += math.sqrt((trajectory_x[j] - trajectory_x[j - 1]) ** 2 + (trajectory_y[j] - trajectory_y[j - 1]) ** 2)
        
        # 计算所需时间
        time = distance / agent_velocities[agent_id]
        
        # 绘制当前智能体（机器人）者的轨迹
        plt.subplot(agent_plt_rows, agent_plt_rows, agent_id // agent_plt_rows * agent_plt_rows + 1 + agent_id % agent_plt_rows)
        plt.title(f"Agent {agent_id}")  # 设置子图标题
        
        # 绘制轨迹线段
        j = 1
        for i in range(len(trajectory_x)):
            if i == 0 or (trajectory_x[i - 1] == trajectory_x[i] and trajectory_y[i - 1] == trajectory_y[i]):
                continue
            # 绘制箭头，表示轨迹的方向
            plt.annotate("", xy=(trajectory_x[i], trajectory_y[i]), xytext=(trajectory_x[i - 1], trajectory_y[i - 1]), 
                         arrowprops=dict(arrowstyle='->', color=colors[agent_id]))
            # 绘制轨迹点的编号
            interpolation_x, interpolation_y = interpolation(trajectory_x[i - 1], trajectory_y[i - 1], trajectory_x[i], trajectory_y[i], 0.333)
            plt.text(interpolation_x, interpolation_y, s=f'{j}')
            j += 1
        
        # 绘制目标位置和智能体（机器人）者的起始位置
        plt.scatter(target_x, target_y, c="black", marker='x')
        plt.scatter(trajectory_x[0], trajectory_y[0], c=colors[agent_id])

    # 绘制所有目标位置的图
    plt.subplot(agent_plt_rows, agent_plt_rows, agent_plt_rows ** 2)
    for i, position in enumerate(target_positions):
        plt.scatter(position.x_, position.y_, c='black', marker='x')  # 绘制目标位置
        plt.text(position.x_ + 4, position.y_ + 4, s=f'{i}')  # 标注目标的编号
    
    plt.show()  # 显示所有绘制的子图


def test():
    # 设置测试参数
    
    # 目标数量
    target_num = 15
    
    # 种群大小，即每一代中的染色体数量
    population_size = 100
    
    # 精英数量，即每一代中保留的最优染色体数量
    elite_num = 4
    
    # 总的迭代次数
    total_iteration = 200
    
    # 每个 Agent 的连续迭代次数
    iteration_unit = 25
    
    # 共享池的大小，即每个 Agent 共享的最优染色体数量
    share_num = 5
    
    # 区域范围，即环境的边长
    area_length = 10000
    
    # 初始化智能体（机器人）的位置列表
    agent_positions: list[Position] = []
    
    # 初始化目标的位置列表
    target_positions: list[Position] = []
    
    # 初始化智能体（机器人）的速度列表
    agent_velocities: list[float] = []
    
    # 设置智能体（机器人）的初始位置
    agent_positions.append(Position(0, 0))  # agent_id: 0
    agent_positions.append(Position(0, area_length))
    agent_positions.append(Position(area_length, area_length))
    agent_positions.append(Position(area_length, 0))
    agent_positions.append(Position(0, area_length / 2))
    agent_positions.append(Position(area_length / 2, area_length))
    agent_positions.append(Position(area_length, area_length / 2))
    agent_positions.append(Position(area_length / 2, 0))  # agent_id: 7
    
    # 设置智能体（机器人）的速度
    agent_velocities.append(70)
    agent_velocities.append(80)
    agent_velocities.append(90)
    agent_velocities.append(100)
    agent_velocities.append(60)
    agent_velocities.append(70)
    agent_velocities.append(80)
    agent_velocities.append(90)
    
    # 设置目标的位置
    for _ in range(target_num):
        # 将每个目标的位置设定在区域的右上角，并标记为目标位置
        target_positions.append(Position(area_length, area_length, True))
    
    # 调用 adaptive_co_evolve 函数进行自适应的协同进化
    # 参数说明：
    # target_num: 目标的数量
    # 3: 目标类型的数量（假设是3）
    # population_size: 种群大小
    # elite_num: 精英数量
    # total_iteration: 总的迭代次数
    # iteration_unit: 每个 Agent 的连续迭代次数
    # [0, 1, 2, 3]: US_list，假设这些值是US类型智能体（机器人）
    # [4, 5, 6, 7]: UA_list，假设这些值是UA类型智能体（机器人）
    # agent_positions: 智能体的位置列表
    # target_positions: 目标的位置列表
    # agent_velocities: 智能体的速度列表
    # share_num: 共享池的大小
    solutions, avg_fitness_list, best_fitness_list = adaptive_co_evolve(target_num, 
                                                                       3, 
                                                                       population_size, 
                                                                       elite_num, 
                                                                       total_iteration, 
                                                                       iteration_unit,
                                                                       [0, 1, 2, 3], 
                                                                       [4, 5, 6, 7], 
                                                                       agent_positions, 
                                                                       target_positions, 
                                                                       agent_velocities,
                                                                       share_num)
    
    # 找到最佳解的智能体（机器人）者索引
    best_agent_index = np.argmin([solution.fitness() for solution in solutions])
    
    # 获取最佳解
    best_solution = solutions[best_agent_index]
    
    # 输出最佳解的智能体（机器人）者索引
    print(f"best solution is from Agent {best_agent_index}")
    
    # 输出最佳解的具体内容
    print(best_solution)
    
    # 输出最佳解的适应度
    print(f"best fitness is {best_solution.fitness()}")
    
    # 绘制所有智能体（机器人）者的平均适应度和最佳适应度变化图
    draw_fitness(avg_fitness_list, best_fitness_list)
    
    # 绘制最佳解的目标分配图
    draw_target_allocation(best_solution, agent_positions, target_positions, agent_velocities)


def test_example():
    # 设置测试参数
    
    # 目标数量
    target_num = 2
    
    # 种群大小，即每一代中的染色体数量
    population_size = 100
    
    # 精英数量，即每一代中保留的最优染色体数量
    elite_num = 4
    
    # 总的迭代次数
    total_iteration = 200
    
    # 每个 Agent 的连续迭代次数
    iteration_unit = 25
    
    # 共享池的大小，即每个 Agent 共享的最优染色体数量
    share_num = 5
    
    # 初始化智能体（机器人）的位置信息
    agent_positions: list[Position] = []
    
    # 初始化目标的位置列表
    target_positions: list[Position] = []
    
    # 初始化智能体（机器人）的速度列表
    agent_velocities: list[float] = []
    
    # 设置智能体（机器人）的初始位置
    # 注意：在实际情况下，所有智能体的位置可能会不同
    agent_positions.append(Position(2500, 0))  # agent_id: 0
    agent_positions.append(Position(2500, 0))  # agent_id: 1
    agent_positions.append(Position(2500, 0))  # agent_id: 2
    
    # 设置智能体（机器人）的速度
    agent_velocities.append(70)  # agent_id: 0 的速度
    agent_velocities.append(80)  # agent_id: 1 的速度
    agent_velocities.append(70)  # agent_id: 2 的速度
    
    # US类型智能体（机器人）列表（US_list）和UA类型智能体（机器人）列表（UA_list）
    US_list = [0, 1]  # US类型智能体（机器人）的 ID 列表
    UA_list = [1, 2]  # UA类型智能体（机器人）的 ID 列表
    
    # 设置目标的位置
    target_positions.append(Position(1000, 3400))  # 目标 0 的位置
    target_positions.append(Position(4500, 4000))  # 目标 1 的位置
    
    # 调用 adaptive_co_evolve 函数进行自适应的协同进化
    # 参数说明：
    # target_num: 目标的数量
    # 3: 目标类型的数量（假设是3）
    # population_size: 种群大小
    # elite_num: 精英数量
    # total_iteration: 总的迭代次数
    # iteration_unit: 每个 Agent 的连续迭代次数
    # US_list: US类型智能体（机器人） ID 列表
    # UA_list: US类型智能体（机器人） ID 列表
    # agent_positions: 智能体的位置列表
    # target_positions: 目标的位置列表
    # agent_velocities: 智能体的速度列表
    # share_num: 共享池的大小
    solutions, avg_fitness_list, best_fitness_list = adaptive_co_evolve(target_num, 
                                                                       3, 
                                                                       population_size, 
                                                                       elite_num, 
                                                                       total_iteration, 
                                                                       iteration_unit,
                                                                       US_list, 
                                                                       UA_list, 
                                                                       agent_positions, 
                                                                       target_positions, 
                                                                       agent_velocities,
                                                                       share_num)
    
    # 找到最佳解的智能体（机器人）者索引
    best_agent_index = np.argmin([solution.fitness() for solution in solutions])
    
    # 获取最佳解
    best_solution = solutions[best_agent_index]
    
    # 输出最佳解的智能体（机器人）者索引
    print(f"best solution is from Agent {best_agent_index}")
    
    # 输出最佳解的具体内容
    print(best_solution)
    
    # 输出最佳解的适应度
    print(f"best fitness is {best_solution.fitness()}")
    
    # 绘制所有智能体（机器人）者的平均适应度和最佳适应度变化图
    draw_fitness(avg_fitness_list, best_fitness_list)
    
    # 绘制最佳解的目标分配图
    draw_target_allocation(best_solution, agent_positions, target_positions, agent_velocities)

# test_example()
test()