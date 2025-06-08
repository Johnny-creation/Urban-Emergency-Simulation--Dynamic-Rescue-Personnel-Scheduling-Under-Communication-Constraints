import random
import math
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from typing import List, Union, Callable


# 模拟任务点
class Task:
    def __init__(self, id, x, y, victims, decline_rate, report_time):
        self.id = id
        self.x = x
        self.y = y
        self.initial_victims = victims
        self.remaining_victims = victims
        self.decline_rate = decline_rate
        self.report_time = report_time
        self.assigned = False
        self.rescued = 0
        self.completed = False
        self.first_arrival_time = None

# 表达式树节点基类
class ExprNode:
    def __init__(self):
        self.parent = None
    
    def evaluate(self, d: float, expected_victims: float, ability: float) -> float:
        raise NotImplementedError
    
    def clone(self) -> 'ExprNode':
        raise NotImplementedError
    
    def __str__(self) -> str:
        raise NotImplementedError

# 常量节点
class ConstNode(ExprNode):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
    
    def evaluate(self, d: float, expected_victims: float, ability: float) -> float:
        return self.value
    
    def clone(self) -> 'ConstNode':
        return ConstNode(self.value)
    
    def __str__(self) -> str:
        return f"{self.value:.2f}"

# 变量节点
class VarNode(ExprNode):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    def evaluate(self, d: float, expected_victims: float, ability: float) -> float:
        if self.name == 'd':
            return max(d, 1e-10)  # 避免除以零
        elif self.name == 'v':
            return max(expected_victims, 0)  # 确保非负
        elif self.name == 'a':
            return ability  # 能力值
        return 0.0
    
    def clone(self) -> 'VarNode':
        return VarNode(self.name)
    
    def __str__(self) -> str:
        return self.name

# 二元操作节点
class BinaryOpNode(ExprNode):
    def __init__(self, op: str, left: ExprNode, right: ExprNode):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right
        left.parent = self
        right.parent = self
    
    def evaluate(self, d: float, expected_victims: float, ability: float) -> float:
        try:
            l_val = self.left.evaluate(d, expected_victims, ability)
            r_val = self.right.evaluate(d, expected_victims, ability)
            
            if self.op == '+':
                return l_val + r_val
            elif self.op == '-':
                return l_val - r_val
            elif self.op == '*':
                return l_val * r_val
            elif self.op == '/':
                if abs(r_val) < 1e-10:
                    return l_val * 1e10  # 处理除以接近零的情况
                return l_val / r_val
            elif self.op == '^':
                # 处理幂运算的特殊情况
                if l_val < 0 and not r_val.is_integer():
                    return 0.0  # 避免复数
                if abs(l_val) < 1e-10 and r_val < 0:
                    return 1e10  # 处理接近零的负幂
                return min(max(l_val ** r_val, -1e10), 1e10)  # 限制范围
            return 0.0
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0  # 处理所有数值错误
    
    def clone(self) -> 'BinaryOpNode':
        return BinaryOpNode(self.op, self.left.clone(), self.right.clone())
    
    def __str__(self) -> str:
        return f"({str(self.left)} {self.op} {str(self.right)})"

# 一元操作节点
class UnaryOpNode(ExprNode):
    def __init__(self, op: str, child: ExprNode):
        super().__init__()
        self.op = op
        self.child = child
        child.parent = self
    
    def evaluate(self, d: float, expected_victims: float, ability: float) -> float:
        try:
            val = self.child.evaluate(d, expected_victims, ability)
            
            if self.op == 'log':
                return math.log(max(abs(val), 1e-10))
            elif self.op == 'exp':
                return min(math.exp(val), 1e10)  # 限制上限
            elif self.op == 'sin':
                return math.sin(val)
            elif self.op == 'cos':
                return math.cos(val)
            elif self.op == 'abs':
                return abs(val)
            elif self.op == 'sqrt':
                return math.sqrt(max(val, 0))
            return val
        except (ValueError, OverflowError):
            return 0.0  # 处理所有数值错误
    
    def clone(self) -> 'UnaryOpNode':
        return UnaryOpNode(self.op, self.child.clone())
    
    def __str__(self) -> str:
        return f"{self.op}({str(self.child)})"

# 生成随机表达式树
def generate_random_expr(max_depth: int = 2) -> ExprNode:
    if max_depth == 0:
        return generate_base_expr()
    
    op_type = random.random()
    if op_type < 0.3:  # 30%概率生成基础表达式
        return generate_base_expr()
    elif op_type < 0.6:  # 30%概率生成简单二元运算
        op = random.choice(['+', '*', '-'])
        return BinaryOpNode(op, 
                          generate_base_expr(),
                          generate_base_expr())
    else:  # 40%概率生成一元运算
        op = random.choice(['sqrt', 'exp', 'log', 'abs'])
        return UnaryOpNode(op, generate_base_expr())

def generate_base_expr() -> ExprNode:
    # 生成基础表达式形式：-d*w1 + v*w2
    epsilon = random.uniform(0.1, 2.0)
    w1 = random.uniform(0.1, 5.0)
    w2 = random.uniform(0.1, 5.0)
    
    neg = BinaryOpNode('*', ConstNode(-1), VarNode('d'))
    first_term = BinaryOpNode('*', neg, ConstNode(w1))
    second_term = BinaryOpNode('*', VarNode('v'), ConstNode(w2))
    return BinaryOpNode('+', first_term, second_term)

# 变异操作
def mutate(node: ExprNode, mutation_rate: float = 0.1) -> ExprNode:
    if random.random() < mutation_rate:
        if isinstance(node, (ConstNode, VarNode)):
            if random.random() < 0.3:  # 30%概率生成基础表达式
                return generate_base_expr()
            else:
                return generate_random_expr(0)
        elif isinstance(node, BinaryOpNode):
            if random.random() < 0.5:
                node.left = mutate(node.left, mutation_rate)
            else:
                node.right = mutate(node.right, mutation_rate)
        elif isinstance(node, UnaryOpNode):
            node.child = mutate(node.child, mutation_rate)
    return node

# 交叉操作
def crossover(parent1: ExprNode, parent2: ExprNode) -> tuple[ExprNode, ExprNode]:
    child1 = parent1.clone()
    child2 = parent2.clone()
    
    # 随机选择交叉点
    nodes1 = []
    nodes2 = []
    
    def collect_nodes(node: ExprNode, nodes: List[ExprNode]):
        nodes.append(node)
        if isinstance(node, BinaryOpNode):
            collect_nodes(node.left, nodes)
            collect_nodes(node.right, nodes)
        elif isinstance(node, UnaryOpNode):
            collect_nodes(node.child, nodes)
    
    collect_nodes(child1, nodes1)
    collect_nodes(child2, nodes2)
    
    if nodes1 and nodes2:
        node1 = random.choice(nodes1)
        node2 = random.choice(nodes2)
        
        if node1.parent and node2.parent:
            if isinstance(node1.parent, BinaryOpNode):
                if node1.parent.left == node1:
                    node1.parent.left = node2
                else:
                    node1.parent.right = node2
            elif isinstance(node1.parent, UnaryOpNode):
                node1.parent.child = node2
            
            if isinstance(node2.parent, BinaryOpNode):
                if node2.parent.left == node2:
                    node2.parent.left = node1
                else:
                    node2.parent.right = node1
            elif isinstance(node2.parent, UnaryOpNode):
                node2.parent.child = node1
    
    return child1, child2

# 救援小组
class RescueGroup:
    def __init__(self, id, ability):
        self.id = id
        self.ability = max(ability, 0.1)
        self.position = (0, 0)
        self.state = 'idle'
        self.current_task = None
        self.time_to_next_state = 0
        self.last_decision_time = 0
        self.score_expr = None
        self.schedule = []

    def score_task(self, task, now):
        d = distance(self.position, (task.x, task.y))
        travel_time = d / self.ability
        effective_time = travel_time + task.report_time + now
        expected_victims = max(0, task.remaining_victims - task.decline_rate * effective_time)
        
        if self.score_expr is None:
            # 默认评分函数，考虑能力值
            return (-d + expected_victims) * self.ability
        
        try:
            score = self.score_expr.evaluate(d, expected_victims, self.ability)
            # 确保评分是有效的数值
            if not np.isfinite(score):
                return (-d + expected_victims) * self.ability
            return score
        except Exception:
            # 如果评分函数出错，使用默认评分
            return (-d + expected_victims) * self.ability

def generate_tasks(n):
    tasks = []
    center_x = random.uniform(20, 80)
    center_y = random.uniform(20, 80)
    for i in range(n):
        x = random.uniform(10, 90)
        y = random.uniform(10, 90)
        victims = random.randint(50, 500)
        decline = random.uniform(0.1, 0.2)
        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        report_time = dist / 5
        tasks.append(Task(i, x, y, victims, decline, report_time))
    return tasks

def generate_groups():
    abilities = np.random.dirichlet(np.ones(5)) * 60
    groups = [RescueGroup(i, round(a, 2)) for i, a in enumerate(abilities)]
    return groups

def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def simulate(groups, tasks, max_time=1000, delay=5):
    time = 0
    while time < max_time and not all(t.completed for t in tasks):
        for group in groups:
            if group.time_to_next_state > 0:
                group.time_to_next_state -= 1
                if group.state == 'working' and group.current_task:
                    rescued = group.ability
                    group.current_task.remaining_victims = max(0, group.current_task.remaining_victims - rescued)
                    group.current_task.rescued += rescued
                    if group.current_task.remaining_victims == 0:
                        group.current_task.completed = True
                        group.state = 'idle'
                        group.current_task = None
                continue

            if group.state == 'moving' and group.current_task:
                group.position = (group.current_task.x, group.current_task.y)
                group.state = 'working'
                group.time_to_next_state = 1
                if group.current_task.first_arrival_time is None:
                    group.current_task.first_arrival_time = time
                continue

            best_score = float('inf')
            best_task = None
            for task in tasks:
                if task.completed:
                    continue
                score = group.score_task(task, time)
                if score < best_score:
                    best_score = score
                    best_task = task

            if best_task and (group.current_task is None or best_task.id != group.current_task.id):
                if time - group.last_decision_time > delay:
                    group.current_task = best_task
                    group.state = 'moving'
                    group.time_to_next_state = math.ceil(distance(group.position, (best_task.x, best_task.y)) / group.ability)
                    group.last_decision_time = time
                    group.schedule.append(best_task.id)

        for t in tasks:
            if not t.completed:
                t.remaining_victims = max(0, t.remaining_victims - t.decline_rate)

        time += 1

def evaluate(groups, tasks, alpha=3, beta=1, gamma=50, delta=3):
    total_rescued = sum(t.rescued for t in tasks)
    overdue = sum(1 for t in tasks if not t.completed and t.remaining_victims > 0)
    times = [len(g.schedule) for g in groups]
    std_dev = np.std(times) if times else 0
    total_response_time = sum((t.first_arrival_time - t.report_time) for t in tasks if t.first_arrival_time)
    fitness = alpha * total_rescued - beta * total_response_time - gamma * overdue - delta * std_dev
    #print(f"Rescued: {total_rescued:.2f} | Response Time: {total_response_time:.2f} | Overdue: {overdue} | Std: {std_dev:.2f} | Fitness: {fitness:.2f}")
    return fitness

# 计算表达式树的深度
def tree_depth(node: ExprNode) -> int:
    if isinstance(node, (ConstNode, VarNode)):
        return 1
    elif isinstance(node, BinaryOpNode):
        left_depth = tree_depth(node.left)
        right_depth = tree_depth(node.right)
        return 1 + max(left_depth, right_depth)
    elif isinstance(node, UnaryOpNode):
        child_depth = tree_depth(node.child)
        return 1 + child_depth
    return 1

# 多次训练，记录能力与权重的映射
ability_records = []
weight_records = []

# 正则化系数
REGULARIZATION_COEFF = 0.1

# 存储最佳表达式
best_exprs = []

def train(task_size, patience=10, pop_size=30, num_environments=100):
    global best_exprs
    random.seed(42)
    np.random.seed(42)
    best_exprs = []  # 重置最佳表达式列表
    best_fitnesses = []  # 存储每个环境的最佳适应度
    
    # 初始化种群，使用generate_base_expr作为初始表达式
    population = [generate_base_expr() for _ in range(pop_size)]
    no_improve = 0
    generation = 0
    best_fitness = -float('inf')
    best_expr = None

    while no_improve < patience:
        scores = []
        for expr in population:
            # 在100个环境中评估每个表达式
            env_scores = []
            for env in range(num_environments):
                tasks = generate_tasks(task_size)
                groups = generate_groups()
                for group in groups:
                    group.score_expr = expr
                simulate(groups, tasks)
                fit = evaluate(groups, tasks)
                
                # 计算正则化项
                depth = tree_depth(expr)
                regularization = REGULARIZATION_COEFF * depth
                
                # 减去正则化项
                fit -= regularization
                env_scores.append(fit)
            
            # 计算平均适应度
            avg_fitness = np.mean(env_scores)
            scores.append((avg_fitness, expr))

        scores.sort(reverse=True, key=lambda x: x[0])
        best = scores[0]
        print(f"Gen {generation} | Best Fitness: {best[0]:.2f}")
        print(f"Best Expression: {str(best[1])}")

        if best[0] > best_fitness:
            best_fitness = best[0]
            best_expr = best[1]
            no_improve = 0
        else:
            no_improve += 1

        # 精英保留
        new_population = [x[1] for x in scores[:pop_size//2]]
        
        # 交叉和变异
        while len(new_population) < pop_size:
            p1, p2 = random.sample(scores[:10], 2)
            child1, child2 = crossover(p1[1], p2[1])
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population[:pop_size]
        generation += 1

    # 在训练结束后，对最佳表达式进行最终评估
    final_scores = []
    for env in range(num_environments):
        tasks = generate_tasks(task_size)
        groups = generate_groups()
        for group in groups:
            group.score_expr = best_expr
        simulate(groups, tasks)
        fit = evaluate(groups, tasks)
        final_scores.append(fit)
    
    final_avg_fitness = np.mean(final_scores)
    print(f"\nFinal Best Expression: {str(best_expr)}")
    print(f"Final Average Fitness: {final_avg_fitness:.2f}")
    
    # 保存最佳表达式
    with open(f"best_expr_{task_size}.txt", "w") as f:
        f.write(f"Best Expression: {str(best_expr)}\n")
        f.write(f"Average Fitness: {final_avg_fitness:.2f}\n")
    
    best_exprs = [best_expr]  # 只保存最佳表达式

def validate(task_size, num_environments=100):
    # 使用固定的随机种子进行验证
    random.seed(42)
    np.random.seed(42)
    
    validation_scores = []
    for env in range(num_environments):
        tasks = generate_tasks(task_size)
        groups = generate_groups()
        
        for group in groups:
            group.score_expr = best_exprs[0]  # 使用最佳表达式
        simulate(groups, tasks)
        fit = evaluate(groups, tasks)
        validation_scores.append(fit)
    
    # 计算平均分数
    avg_score = np.mean(validation_scores)
    print("\nValidation Results:")
    print(f"Expression: {str(best_exprs[0])}")
    print(f"Average Fitness: {avg_score:.2f}\n")

def test(task_size, num_environments=100):
    # 使用固定的随机种子进行测试
    random.seed(42)
    np.random.seed(42)
    
    test_scores = []
    for env in range(num_environments):
        tasks = generate_tasks(task_size)
        groups = generate_groups()
        
        for group in groups:
            group.score_expr = best_exprs[0]  # 使用最佳表达式
        simulate(groups, tasks)
        fit = evaluate(groups, tasks)
        test_scores.append(fit)
    
    # 计算平均分数
    avg_score = np.mean(test_scores)
    print("\nTest Results:")
    print(f"Expression: {str(best_exprs[0])}")
    print(f"Average Fitness: {avg_score:.2f}\n")

if __name__ == "__main__":
    for label, size in zip(["small","medium", "large"], [ 15,50, 200]):
        print(f"\nTraining on {label} task set...")
        train(size, num_environments=1000)
        print(f"\nValidating on {label} task set...")
        validate(size, num_environments=1000)
        print(f"\nTesting on {label} task set...")
        test(size, num_environments=1000)