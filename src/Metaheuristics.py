# 灾难救援元启发式调度 + 遗传算法优化评分函数（含三态、可抢占 + 拟合能力-权重关系）
import random
import math
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge


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
        self.w1 = 0.0
        self.w2 = 0.0
        self.schedule = []

    def score_task(self, task, now):
        d = distance(self.position, (task.x, task.y))
        travel_time = d / self.ability
        effective_time = travel_time + task.report_time + now
        expected_victims = max(0, task.remaining_victims - task.decline_rate * effective_time)
        return self.w1 * d + self.w2 * expected_victims

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

def evaluate(groups, tasks, alpha=1, beta=1, gamma=1, delta=3):
    total_rescued = sum(t.rescued for t in tasks)
    overdue = sum(1 for t in tasks if not t.completed and t.remaining_victims > 0)
    times = [len(g.schedule) for g in groups]
    std_dev = np.std(times) if times else 0
    total_response_time = sum((t.first_arrival_time - t.report_time) for t in tasks if t.first_arrival_time)
    fitness = alpha * total_rescued - beta * total_response_time - gamma * overdue - delta * std_dev
    print(f"Rescued: {total_rescued:.2f} | Response Time: {total_response_time:.2f} | Overdue: {overdue} | Std: {std_dev:.2f} | Fitness: {fitness:.2f}")
    return fitness

# 多次训练，记录能力与权重的映射
ability_records = []
weight_records = []

def train(task_size, patience=10, pop_size=30, repeats=10):
    global ability_records, weight_records
    best_weights_all = []
    for r in range(repeats):
        tasks_template = generate_tasks(task_size)
        best_weights = None
        best_fitness = -float('inf')
        population = [np.random.uniform(0, 1, 10) for _ in range(pop_size)]
        no_improve = 0
        generation = 0

        while no_improve < patience:
            scores = []
            for p in population:
                tasks = [Task(t.id, t.x, t.y, t.initial_victims, t.decline_rate, t.report_time) for t in tasks_template]
                groups = generate_groups()
                for i, group in enumerate(groups):
                    group.w1 = p[i*2]
                    group.w2 = p[i*2+1]
                simulate(groups, tasks)
                fit = evaluate(groups, tasks)
                scores.append((fit, p, [g.ability for g in groups]))

            scores.sort(reverse=True, key=lambda x: x[0])
            best = scores[0]
            print(f"[Run {r}] Gen {generation} | Best Fitness: {best[0]:.2f}")

            if best[0] > best_fitness:
                best_fitness = best[0]
                best_weights = best[1]
                best_abilities = best[2]
                no_improve = 0
            else:
                no_improve += 1

            population = [x[1] for x in scores[:pop_size//2]]
            while len(population) < pop_size:
                p1, p2 = random.sample(population[:10], 2)
                cross = np.where(np.random.rand(10) > 0.5, p1, p2)
                mutation = np.random.uniform(-0.1, 0.1, 10)
                child = np.clip(cross + mutation, 0, 1)
                population.append(child)

            generation += 1

        # 记录能力与权重的对应
        for i in range(5):
            ability_records.append(best_abilities[i])
            weight_records.append((best_weights[i*2], best_weights[i*2+1]))
        best_weights_all.append(best_weights)
    return best_weights_all

def fit_ability_to_weight():
    abilities = np.array(ability_records).reshape(-1, 1)
    w1s = np.array([w[0] for w in weight_records])
    w2s = np.array([w[1] for w in weight_records])
    # model1 = LinearRegression().fit(abilities, w1s)
    # model2 = LinearRegression().fit(abilities, w2s)
    model1 = Ridge(alpha=1e-3).fit(abilities, w1s)
    model2 = Ridge(alpha=1e-3).fit(abilities, w2s)

    output = (
        f"===== 回归结果（task_size={size}） =====\n"
        f"w1 ≈ {model1.coef_[0]:.4f} * ability + {model1.intercept_:.4f}\n"
        f"w2 ≈ {model2.coef_[0]:.4f} * ability + {model2.intercept_:.4f}\n"
    )
    print("\n" + output)
    with open(f"regression_result_{size}.txt", "w") as f:
        f.write(output)
    return model1, model2

def save_weights(weights, size_label):
    with open(f"weights_{size_label}.pkl", "wb") as f:
        pickle.dump(weights, f)

def test(task_size, model1, model2):
    tasks = generate_tasks(task_size)
    groups = generate_groups()
    for group in groups:
        group.w1 = model1.predict([[group.ability]])[0]
        group.w2 = model2.predict([[group.ability]])[0]
    simulate(groups, tasks)
    fit = evaluate(groups, tasks)
    print(f"Test fitness (task size {task_size}):", fit)
    return fit

if __name__ == "__main__":
    for label, size in zip(["small", "medium", "large"], [10, 50, 200]):
        print(f"Training on {label} task set...")
        train(size)
        print("Fitting ability-weight mapping...")
        m1, m2 = fit_ability_to_weight()
        print(f"Testing on {label} task set...")
        test(size, m1, m2)
