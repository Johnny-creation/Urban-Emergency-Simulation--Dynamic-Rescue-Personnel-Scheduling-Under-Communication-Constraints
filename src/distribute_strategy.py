import random
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from collections import defaultdict
# 设置为Windows系统中的中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号



class Task:
    def __init__(self, id: int, x: float, y: float, initial_victims: int, decline_rate: float, report_time: float):
        self.id = id
        self.x = x
        self.y = y
        self.initial_victims = initial_victims
        self.current_victims = initial_victims
        # 无救援时衰减率
        self.decline_rate = decline_rate
        # 有救援时的衰减率（假设救援介入后减半）
        self.rescue_decline_rate = decline_rate * 0.5
        self.report_time = report_time
        self.reported = False

class Rescuer:
    def __init__(self, id: int, x: float, y: float, rescuer_type: str):
        self.id = id
        self.position = {"x": x, "y": y}
        self.state = "IDLE"
        self.current_task_id = None
        self.type = rescuer_type

class Simulation:
    def __init__(self, scenario: str):
        self.scenario = scenario
        self.tasks: List[Task] = []
        self.rescue_center_position = {"x": 0, "y": 0}
        self.total_initial_victims = 0
        self.current_time = 0
        self.initialize_scenario()

    def initialize_scenario(self):
        task_counts = {
            'small': 10,
            'medium': 50,
            'large': 200
        }
        
        num_tasks = task_counts[self.scenario]
        self.tasks = []
        
        # 使用当前时间戳作为随机种子
        random.seed()
        
        # 随机生成救援中心位置
        center_x = random.random() * 60 + 20
        center_y = random.random() * 60 + 20
        self.rescue_center_position = {"x": center_x, "y": center_y}
        
        for i in range(num_tasks):
            # 随机生成灾情点位置
            while True:
                x = random.random() * 80 + 10;
                y = random.random() * 80 + 10;
                if math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) >= 5:
                    break
            
            # 距离越远，报告时间越长
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            report_time = distance / 5
            
            # 随机生成受灾人数 (50-500)
            victims = random.randint(50, 500)
            
            # 随机衰减率 (0.1-0.2)
            decline_rate = random.uniform(0.1, 0.2)
            
            self.tasks.append(Task(i, x, y, victims, decline_rate, report_time))
            self.total_initial_victims += victims

    def get_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        return math.sqrt((pos1["x"] - pos2["x"]) ** 2 + (pos1["y"] - pos2["y"]) ** 2)

class NearestFirstSimulation(Simulation):
    def __init__(self, scenario: str):
        super().__init__(scenario)
        self.rescuer_position = {"x": self.rescue_center_position["x"], "y": self.rescue_center_position["y"]}
        self.rescuer_state = "IDLE"
        self.current_task_id = None
        self.rescued = 0
        self.completed = False
        self.completion_time = None

    def update(self, time_step: float = 0.1):
        self.current_time += time_step
        
        # 更新任务状态
        for task in self.tasks:
            if not task.reported and self.current_time >= task.report_time:
                task.reported = True
            
            if task.reported and task.current_victims > 0:
            # 判断救援中干预：在 Single-Agent 中，看当前 rescuer_state & current_task_id
            # 在 Multi-Agent 中，看是否有任一 rescuer.state=="AT_TASK" 且 current_task_id==task.id
                being_rescued = False
                if hasattr(self, 'rescuer_state'):
                    being_rescued = (
                        self.rescuer_state == "AT_TASK"
                        and self.current_task_id == task.id
                    )
           # --- Multi-Agent 示例:
                if hasattr(self, 'rescuers'):
                    being_rescued = any(
                        r.state == "AT_TASK" and r.current_task_id == task.id 
                        for r in self.rescuers
                )
            # 根据是否被救援选择衰减率
                rate = task.rescue_decline_rate if being_rescued else task.decline_rate
                decline = task.initial_victims * rate * time_step / 60
                task.current_victims = max(0, task.current_victims - decline)
        
        # 更新救援人员状态
        if self.rescuer_state == "IDLE":
            available_tasks = [t for t in self.tasks if t.reported and t.current_victims > 0]
            if available_tasks:
                selected_task = min(available_tasks, 
                                  key=lambda t: self.get_distance(self.rescuer_position, {"x": t.x, "y": t.y}))
                self.current_task_id = selected_task.id
                self.rescuer_state = "MOVING_TO_TASK"
        
        elif self.rescuer_state == "MOVING_TO_TASK":
            task = next((t for t in self.tasks if t.id == self.current_task_id), None)
            if task:
                distance = self.get_distance(self.rescuer_position, {"x": task.x, "y": task.y})
                if distance < 1:
                    self.rescuer_position = {"x": task.x, "y": task.y}
                    self.rescuer_state = "AT_TASK"
                else:
                    move_speed = 1
                    dx = task.x - self.rescuer_position["x"]
                    dy = task.y - self.rescuer_position["y"]
                    self.rescuer_position["x"] += (dx / distance) * move_speed
                    self.rescuer_position["y"] += (dy / distance) * move_speed
        
        elif self.rescuer_state == "AT_TASK":
            task = next((t for t in self.tasks if t.id == self.current_task_id), None)
            if task:
                if task.current_victims <= 0:
                    self.current_task_id = None
                    self.rescuer_state = "IDLE"
                else:
                    max_possible_rescue = min(task.current_victims, 30)
                    actual_rescued = min(max_possible_rescue, self.total_initial_victims - self.rescued)
                    if actual_rescued > 0:
                        self.rescued += actual_rescued
                        task.current_victims = max(0, task.current_victims - actual_rescued)
        
        # 检查是否完成
        if not self.completed and all(task.current_victims <= 0 for task in self.tasks):
            self.completed = True
            self.completion_time = self.current_time

class LargestFirstSimulation(Simulation):
    def __init__(self, scenario: str):
        super().__init__(scenario)
        self.rescuer_position = {"x": self.rescue_center_position["x"], "y": self.rescue_center_position["y"]}
        self.rescuer_state = "IDLE"
        self.current_task_id = None
        self.rescued = 0
        self.completed = False
        self.completion_time = None

    def update(self, time_step: float = 0.1):
        self.current_time += time_step
        
        # 更新任务状态
        for task in self.tasks:
            if not task.reported and self.current_time >= task.report_time:
                task.reported = True
            
            if task.reported and task.current_victims > 0:
            # 判断救援中干预：在 Single-Agent 中，看当前 rescuer_state & current_task_id
            # 在 Multi-Agent 中，看是否有任一 rescuer.state=="AT_TASK" 且 current_task_id==task.id
                being_rescued = False
                if hasattr(self, 'rescuer_state'):
                    being_rescued = (
                        self.rescuer_state == "AT_TASK"
                        and self.current_task_id == task.id
                    )
           # --- Multi-Agent 示例:
                if hasattr(self, 'rescuers'):
                    being_rescued = any(
                        r.state == "AT_TASK" and r.current_task_id == task.id 
                        for r in self.rescuers
                )
            # 根据是否被救援选择衰减率
                rate = task.rescue_decline_rate if being_rescued else task.decline_rate
                decline = task.initial_victims * rate * time_step / 60
                task.current_victims = max(0, task.current_victims - decline)

        # 更新救援人员状态
        if self.rescuer_state == "IDLE":
            available_tasks = [t for t in self.tasks if t.reported and t.current_victims > 0]
            if available_tasks:
                selected_task = max(available_tasks, key=lambda t: t.current_victims)
                self.current_task_id = selected_task.id
                self.rescuer_state = "MOVING_TO_TASK"
        
        elif self.rescuer_state == "MOVING_TO_TASK":
            task = next((t for t in self.tasks if t.id == self.current_task_id), None)
            if task:
                distance = self.get_distance(self.rescuer_position, {"x": task.x, "y": task.y})
                if distance < 1:
                    self.rescuer_position = {"x": task.x, "y": task.y}
                    self.rescuer_state = "AT_TASK"
                else:
                    move_speed = 1
                    dx = task.x - self.rescuer_position["x"]
                    dy = task.y - self.rescuer_position["y"]
                    self.rescuer_position["x"] += (dx / distance) * move_speed
                    self.rescuer_position["y"] += (dy / distance) * move_speed
        
        elif self.rescuer_state == "AT_TASK":
            task = next((t for t in self.tasks if t.id == self.current_task_id), None)
            if task:
                if task.current_victims <= 0:
                    self.current_task_id = None
                    self.rescuer_state = "IDLE"
                else:
                    max_possible_rescue = min(task.current_victims, 30)
                    actual_rescued = min(max_possible_rescue, self.total_initial_victims - self.rescued)
                    if actual_rescued > 0:
                        self.rescued += actual_rescued
                        task.current_victims = max(0, task.current_victims - actual_rescued)
        
        # 检查是否完成
        if not self.completed and all(task.current_victims <= 0 for task in self.tasks):
            self.completed = True
            self.completion_time = self.current_time

# 在MultiAgentSimulation类中添加策略统计
MultiAgentSimulation_strategy_counts = {
    'NEAREST': 0,
    'LARGEST': 0,
    'HYBRID': 0, 
}


def allocate_rescuers(total_rescuers: int = 15, 
                     total_capacity: int = 30, 
                     min_groups: int = 3, 
                     max_groups: int = 8) -> List[Dict]:
    
    # 修改分组数量生成逻辑，增加中间值的概率
    group_options = list(range(min_groups, max_groups+1))
    # 使用三角形分布（中间值概率更高）
    num_groups = int(random.triangular(min_groups, max_groups, (min_groups + max_groups)/2))
    
    # 保证在合法范围内
    num_groups = max(min_groups, min(num_groups, max_groups))
    
    # 初始化每组至少有1人
    groups = [1 for _ in range(num_groups)]
    remaining = total_rescuers - num_groups
    
    # 随机分配剩余的人数
    for _ in range(remaining):
        idx = random.randint(0, num_groups-1)
        groups[idx] += 1
    
    # 3. 随机分配能力值 (总和为30)
    # 初始化每组至少有1点能力值
    capacities = [1 for _ in range(num_groups)]
    remaining_cap = total_capacity - num_groups
    
    # 随机分配剩余的能力值
    for _ in range(remaining_cap):
        idx = random.randint(0, num_groups-1)
        capacities[idx] += 1
    
    # 4. 打乱能力值分配，使其与人数不成正比
    random.shuffle(capacities)
    
    # 5. 组合结果
    result = []
    for size, cap in zip(groups, capacities):
        result.append({
            'size': size,
            'capacity': cap
        })
    
    return result

# 在MultiAgentSimulation类中修改初始化方法
class MultiAgentSimulation(Simulation):
    def __init__(self, scenario: str):
        super().__init__(scenario)
        
         # 新分组分配逻辑
        groups = allocate_rescuers()  # 使用新分配函数
        capacities = [g['capacity'] for g in groups]
        sizes = [g['size'] for g in groups]
        
        # 策略分配（允许重复）
        strategies = random.choices(['NEAREST', 'LARGEST', 'HYBRID'], k=len(groups))
        
        # 构建分组配置
        self.group_config = []
        for cap, size, strategy in zip(capacities, sizes, strategies):
            self.group_config.append({
                'capacity': cap,
                'size': size,
                'strategy': strategy
            })
        
        # 创建救援者
        self.rescuers = []
        rescuer_id = 1
        for group in self.group_config:
            for _ in range(group['size']):
                rescuer = Rescuer(rescuer_id, self.rescue_center_position["x"],
                                self.rescue_center_position["y"], group['strategy'])
                rescuer.capacity = group['capacity']
                self.rescuers.append(rescuer)
                rescuer_id += 1

        # 初始化救援人数统计
        self.rescued = 0  # 添加这一行
        self.strategy_counts = {
            'NEAREST': sum(1 for r in self.rescuers if r.type == 'NEAREST'),
            'LARGEST': sum(1 for r in self.rescuers if r.type == 'LARGEST'),
            'HYBRID': sum(1 for r in self.rescuers if r.type == 'HYBRID')
        }
        MultiAgentSimulation_strategy_counts["NEAREST"]=self.strategy_counts["NEAREST"]
        MultiAgentSimulation_strategy_counts["LARGEST"]=self.strategy_counts["LARGEST"]
        MultiAgentSimulation_strategy_counts["HYBRID"]=self.strategy_counts["HYBRID"]
        self.completed = False
        self.completion_time = None
        self.task_assignments = {}

    def update(self, time_step: float = 0.1):
        self.current_time += time_step
        rescued_this_step = 0
        # 添加距离缓存优化
        distance_cache = {}
        
        def get_cached_distance(pos1, pos2):
            key = (id(pos1), id(pos2))
            if key not in distance_cache:
                distance_cache[key] = self.get_distance(pos1, pos2)
            return distance_cache[key]
        
        # 在任务选择中使用缓存距离
        for rescuer in self.rescuers:
            if rescuer.state == "IDLE":
                available_tasks = [t for t in self.tasks if t.reported and t.current_victims > 0]
                if available_tasks:
                    if rescuer.type == "NEAREST":
                        selected_task = min(available_tasks, 
                                          key=lambda t: get_cached_distance(rescuer.position, {"x": t.x, "y": t.y}))

        
        
        # Update task states
        for task in self.tasks:
            if not task.reported and self.current_time >= task.report_time:
                task.reported = True
            
            if task.reported and task.current_victims > 0:
            # 判断救援中干预：在 Single-Agent 中，看当前 rescuer_state & current_task_id
            # 在 Multi-Agent 中，看是否有任一 rescuer.state=="AT_TASK" 且 current_task_id==task.id
                being_rescued = False
                if hasattr(self, 'rescuer_state'):
                    being_rescued = (
                        self.rescuer_state == "AT_TASK"
                        and self.current_task_id == task.id
                    )
           # --- Multi-Agent 示例:
                if hasattr(self, 'rescuers'):
                    being_rescued = any(
                        r.state == "AT_TASK" and r.current_task_id == task.id 
                        for r in self.rescuers
                )
            # 根据是否被救援选择衰减率
                rate = task.rescue_decline_rate if being_rescued else task.decline_rate
                decline = task.initial_victims * rate * time_step / 60
                task.current_victims = max(0, task.current_victims - decline)
                
        # Update rescuers
        for rescuer in self.rescuers:
            if rescuer.state == "IDLE":
                available_tasks = [t for t in self.tasks if t.reported and t.current_victims > 0 
                                 and t.id not in self.task_assignments.values()]
                if available_tasks:
                    if rescuer.type == "NEAREST":
                        selected_task = min(available_tasks, 
                                          key=lambda t: self.get_distance(rescuer.position, {"x": t.x, "y": t.y}))
                    elif rescuer.type == "LARGEST":
                        selected_task = max(available_tasks, key=lambda t: t.current_victims)
                    else:  # HYBRID
                        selected_task = max(available_tasks, 
                                          key=lambda t: t.current_victims / 
                                          (self.get_distance(rescuer.position, {"x": t.x, "y": t.y}) + 1))
                    
                    self.task_assignments[rescuer.id] = selected_task.id
                    rescuer.current_task_id = selected_task.id
                    rescuer.state = "MOVING_TO_TASK"
            
            elif rescuer.state == "MOVING_TO_TASK":
                task = next((t for t in self.tasks if t.id == rescuer.current_task_id), None)
                if task:
                    distance = self.get_distance(rescuer.position, {"x": task.x, "y": task.y})
                    if distance < 1:
                        rescuer.position = {"x": task.x, "y": task.y}
                        rescuer.state = "AT_TASK"
                    else:
                        move_speed = 1
                        dx = task.x - rescuer.position["x"]
                        dy = task.y - rescuer.position["y"]
                        rescuer.position["x"] += (dx / distance) * move_speed
                        rescuer.position["y"] += (dy / distance) * move_speed
            
            elif rescuer.state == "AT_TASK":
                task = next((t for t in self.tasks if t.id == rescuer.current_task_id), None)
                if task:
                    if task.current_victims <= 0:
                        if rescuer.id in self.task_assignments:
                            del self.task_assignments[rescuer.id]
                        rescuer.current_task_id = None
                        rescuer.state = "IDLE"
                    else:
                        max_possible_rescue = min(task.current_victims, rescuer.capacity)
                        actual_rescued = min(max_possible_rescue, 
                                           self.total_initial_victims - (self.rescued + rescued_this_step))
                        if actual_rescued > 0:
                            rescued_this_step += actual_rescued
                            task.current_victims = max(0, task.current_victims - actual_rescued)
        
        self.rescued = min(self.rescued + rescued_this_step, self.total_initial_victims)
        
        # Check completion
        if not self.completed and all(task.current_victims <= 0 for task in self.tasks):
            self.completed = True
            self.completion_time = self.current_time

# 修改run_simulation的返回元组，增加分组配置
def run_simulation(scenario: str) -> Tuple[
    float,            # nearest success rate
    float,            # nearest completion time
    int,              # nearest rescued
    float,            # largest success rate
    float,            # largest completion time
    int,              # largest rescued
    float,            # multi-agent success rate
    float,            # multi-agent completion time
    int,              # multi-agent rescued
    int,              # count NEAREST
    int,              # count LARGEST
    int,              # count HYBRID
    List[Dict[str, int]]  # group_config 列表
]:
    nearest = NearestFirstSimulation(scenario)
    largest = LargestFirstSimulation(scenario)
    multi_agent = MultiAgentSimulation(scenario)
    
    # 优化为各策略独立运行（原代码同时运行三种策略）
    def run_single(sim):
        while not sim.completed and sim.current_time < 300:
            sim.update()
        return sim
    
    nearest = run_single(nearest)
    largest = run_single(largest) 
    multi_agent = run_single(multi_agent)
    
    
    # while not (nearest.completed and largest.completed and multi_agent.completed) and \
    #       nearest.current_time < 300 and largest.current_time < 300 and multi_agent.current_time < 300:
    #     nearest.update()
    #     largest.update()
    #     multi_agent.update()
    
    
    return(
        nearest.rescued / nearest.total_initial_victims,  # 0: nearest success
        nearest.completion_time or 300,                  # 1: nearest time
        nearest.rescued,                                 # 2: nearest rescued
        largest.rescued / largest.total_initial_victims, # 3: largest success
        largest.completion_time or 300,                  # 4: largest time
        largest.rescued,                                # 5: largest rescued
        multi_agent.rescued / multi_agent.total_initial_victims, #6: multi success
        multi_agent.completion_time or 300,             #7: multi time
        multi_agent.rescued,                            #8: multi rescued
        multi_agent.strategy_counts['NEAREST'],         #9: count NEAREST
        multi_agent.strategy_counts['LARGEST'],         #10: count LARGEST
        multi_agent.strategy_counts['HYBRID'],          #11: count HYBRID
        multi_agent.group_config 
        
    )

# 修改main函数中的结果收集
def main():
    scenarios = ['small', 'medium', 'large']
    num_iterations = 10
    analysis_results = {}  # 用于存储分析结果
    
    for scenario in scenarios:
        # 修改默认字典初始化
        config_stats = defaultdict(lambda: {
            'success_rates': [],
            'config_details': None,
            'success_sum': 0,
            'count': 0,
            'config': [],
            'total_rescued': 0 
        })
        
        print(f"\n运行 {scenario} 场景的 {num_iterations} 次模拟...")
        
        
         # 修改分析部分
        ranked_configs = []
        for key, data in config_stats.items():
            avg_success = np.mean(data['success_rates']) * 100  # 现在可以正常访问
            ranked_configs.append((avg_success, data['config_details']))
        
        # 按成功率降序排序
        ranked_configs.sort(reverse=True, key=lambda x: x[0])
        analysis_results[scenario] = ranked_configs[:3]

        
        
       # 修改后的最佳记录变量
        best_record = {
            'success_rate': 0,
            'completion_time': 300,
            'rescued': 0,
            'strategy': {'NEAREST': 0, 'LARGEST': 0, 'HYBRID': 0},
            'group_config': []
        }
        
        # 存储每次模拟的结果
        nearest_success_rates = []
        nearest_times = []
        nearest_rescued = []
        largest_success_rates = []
        largest_times = []
        largest_rescued = []
        multi_agent_success_rates = []
        multi_agent_times = []
        multi_agent_rescued = []
        multi_agent_strategies = []  # 新增策略记录列表
        
        
        for i in range(num_iterations):
            if (i + 1) % 10 == 0:
                print(f"完成 {i + 1} 次模拟")
            
            
            
            # 收集三种策略的数据
            # collect_data(nearest, largest, multi_agent)  # 需要实现数据收集函数
        
            
            results = run_simulation(scenario)
            
            group_config = results[12]
            config_key = tuple(sorted(
                (f"g{len(group_config)}c{g['capacity']}s{g['size']}{g['strategy'][:2]}" 
                 for g in group_config),
                key=lambda x: (-int(x.split('c')[1].split('s')[0]), -int(x.split('s')[1][:1]))
            ))
            
            # 统一更新配置统计（保持键一致）
            config_stats[config_key]['success_rates'].append(results[6])
            config_stats[config_key]['total_rescued'] += results[8]
            config_stats[config_key]['success_sum'] += results[6]  # 添加缺失的键
            config_stats[config_key]['count'] += 1
            config_stats[config_key]['config_details'] = group_config


        # 新增配置分析部分
        ranked_configs = []
        for config_key, data in config_stats.items():
            avg_success = np.mean(data['success_rates']) * 100
            avg_rescued = data['total_rescued'] / len(data['success_rates'])
            ranked_configs.append( (avg_success, avg_rescued, data['config_details']) )
        
        # 按成功率排序取前三
        ranked_configs.sort(reverse=True, key=lambda x: (x[0], x[1]))
        top_configs = ranked_configs[:3]
        analysis_results[scenario] = top_configs
        
        # 输出分析结果
        print(f"\n{scenario}场景最优配置排名:")
        for rank, (success, rescued, config) in enumerate(top_configs, 1):
            print(f"\n第{rank}名: 平均成功率 {success:.1f}% 平均救援 {rescued:.0f}人")
            for i, group in enumerate(config, 1):
                print(f"组{i}: {group['size']}人 能力{group['capacity']} 策略:{group['strategy']}")
        
    
            
        #      # 修改后的数据收集
        #     current_success = results[6]
        #     current_record = {
        #         'success_rate': current_success,
        #         'completion_time': results[7],
        #         'rescued': results[8],
        #         'strategy': {
        #             'NEAREST': results[9],
        #             'LARGEST': results[10],
        #             'HYBRID': results[11]
        #         },
        #         'group_config': results[12]  # 获取新增的第13个返回参数
        #     }
            
            
        #    # 更新最佳记录
        #     if current_record['success_rate'] > best_record['success_rate']:
        #         best_record = current_record

        
        # 计算平均值
        # avg_nearest_success = np.mean(nearest_success_rates) * 100
        # avg_nearest_time = np.mean(nearest_times)
        # avg_nearest_rescued = np.mean(nearest_rescued)
        # avg_largest_success = np.mean(largest_success_rates) * 100
        # avg_largest_time = np.mean(largest_times)
        # avg_largest_rescued = np.mean(largest_rescued)
        # avg_multi_agent_success = np.mean(multi_agent_success_rates) * 100
        # avg_multi_agent_time = np.mean(multi_agent_times)
        # avg_multi_agent_rescued = np.mean(multi_agent_rescued) 
        
        
        # print(f"\n{scenario} 场景的平均结果:")
        # print(f"最近任务优先: 成功率 {avg_nearest_success:.1f}%, 完成时间 {avg_nearest_time:.1f}分钟, 救援人数 {avg_nearest_rescued:.0f}")
        # print(f"最大任务优先: 成功率 {avg_largest_success:.1f}%, 完成时间 {avg_largest_time:.1f}分钟, 救援人数 {avg_largest_rescued:.0f}")
        # print(f"多智能体策略: 成功率 {avg_multi_agent_success:.1f}%, 完成时间 {avg_multi_agent_time:.1f}分钟, "
        #       f"救援人数 {avg_multi_agent_rescued:.0f}")
        
        # # 修改后的最佳结果输出
        # print(f"\n最佳单次结果 - 每组配置:")
        # for i, group in enumerate(best_record['group_config'], 1):
        #     print(f"组{i}: 人数={group['size']}人, 策略={group['strategy']}, 能力={group['capacity']}人/分钟")
        # print(f"完成时间: {best_record['completion_time']:.1f}分钟")
        # print(f"救援人数: {best_record['rescued']:.0f}") 
        # print(f"策略分布 - 最近: {best_record['strategy']['NEAREST']}人, "
        #       f"最大: {best_record['strategy']['LARGEST']}人, "
        #       f"混合: {best_record['strategy']['HYBRID']}人")
        
        
        
        # # 绘制柱状图
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # # 成功率对比
        # success_rates = [avg_nearest_success, avg_largest_success, avg_multi_agent_success]
        # ax1.bar(['最近任务优先', '最大任务优先', '多智能体策略'], success_rates)
        # ax1.set_title('成功率对比 (%)')
        # ax1.set_ylim(0, 100)
        
        # # 完成时间对比
        # completion_times = [avg_nearest_time, avg_largest_time, avg_multi_agent_time]
        # ax2.bar(['最近任务优先', '最大任务优先', '多智能体策略'], completion_times)
        # ax2.set_title('完成时间对比 (分钟)')
        
        # # 救援人数对比
        # rescued_numbers = [avg_nearest_rescued, avg_largest_rescued, avg_multi_agent_rescued]
        # ax3.bar(['最近任务优先', '最大任务优先', '多智能体策略'], rescued_numbers)
        # ax3.set_title('救援人数对比')
        
        # plt.tight_layout()
        # plt.savefig(f'{scenario}_comparison.png')
        # plt.close()
        
    # 打印结果
        # print(f"\n{scenario}场景最优配置:")
        # for rank, (success, rescued, config) in enumerate(top_configs, 1):
        #     print(f"\n第{rank}名: 平均成功率 {success:.1f}% 平均救援 {rescued:.0f}人")
        #     print(f"总组数: {len(config)}")
        #     for i, group in enumerate(config, 1):
        #         print(f"组{i}: {group['size']}人 | 能力:{group['capacity']} | 策略:{group['strategy']}")

    return analysis_results


if __name__ == "__main__":
    main() 