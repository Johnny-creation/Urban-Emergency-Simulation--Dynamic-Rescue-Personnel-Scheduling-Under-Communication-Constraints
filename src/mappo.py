import random
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os

# 设置为Windows系统中的中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


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
        self.ability = max(ability, 1)
        self.position = (0, 0)
        self.state = 'idle'
        self.current_task = None

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


class EmergencyEnv:
    def __init__(self, num_tasks=50, num_groups=5, seed=None):
        self.num_tasks = num_tasks
        self.num_groups = num_groups
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.tasks = generate_tasks(num_tasks)
        self.groups = generate_groups()
        self.current_time = 0
        self.max_time = 1000
        self.delay = 5
        self.total_victims = sum(task.initial_victims for task in self.tasks)
        self.rescued_victims = 0
        self.last_rescued = 0  # 用于计算增量奖励
        
    def reset(self):
        self.tasks = generate_tasks(self.num_tasks)
        self.groups = generate_groups()
        self.current_time = 0
        self.total_victims = sum(task.initial_victims for task in self.tasks)
        self.rescued_victims = 0
        self.last_rescued = 0
        return self._get_observation()
    
    def _get_observation(self):
        # 构建观察空间
        obs = []
        for group in self.groups:
            group_obs = [
                group.position[0] / 100,  # 归一化坐标
                group.position[1] / 100,
                group.ability / 60,  # 归一化能力值
                1.0 if group.state == 'idle' else 0.0,  # 状态编码
                self.current_time / self.max_time  # 时间信息
            ]
            
            # 添加任务信息
            for task in self.tasks:
                task_obs = [
                    task.x / 100,  # 归一化坐标
                    task.y / 100,
                    task.remaining_victims / 500,  # 归一化剩余人数
                    task.decline_rate / 0.2,  # 归一化衰减率
                    1.0 if task.assigned else 0.0,  # 是否被分配
                    task.first_arrival_time / self.max_time if task.first_arrival_time else 0.0  # 首次到达时间
                ]
                group_obs.extend(task_obs)
            
            obs.append(group_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions):
        # 执行动作
        rewards = []
        for i, (group, action) in enumerate(zip(self.groups, actions)):
            if action < self.num_tasks:  # 选择任务
                task = self.tasks[action]
                if not task.assigned:
                    task.assigned = True
                    task.first_arrival_time = self.current_time
                    group.current_task = task
                    group.state = 'busy'
                    
                    # 计算任务优先级
                    priority = (task.remaining_victims / self.total_victims) * (1 + group.ability / 60)
                    # 考虑距离因素
                    dist = distance(group.position, (task.x, task.y))
                    dist_factor = 1.0 / (1.0 + dist/100)  # 归一化距离影响
                    
                    # 综合奖励
                    reward = priority * dist_factor
                    rewards.append(reward)
                else:
                    rewards.append(-0.2)  # 惩罚重复选择
            else:
                rewards.append(-0.1)  # 惩罚不选择任务
        
        # 更新环境
        self.current_time += 1
        
        # 更新任务状态
        for task in self.tasks:
            if task.assigned and not task.completed:
                old_victims = task.remaining_victims
                task.remaining_victims = max(0, task.remaining_victims - 
                    task.decline_rate * task.initial_victims)
                if task.remaining_victims == 0:
                    task.completed = True
                # 计算成功救援的人数
                rescued = old_victims - task.remaining_victims
                self.rescued_victims += rescued
        
        # 计算增量奖励
        rescue_increment = self.rescued_victims - self.last_rescued
        self.last_rescued = self.rescued_victims
        
        # 计算总奖励
        total_reward = sum(rewards)
        
        # 添加全局奖励
        rescue_ratio = self.rescued_victims / self.total_victims
        total_reward += rescue_increment * 5  # 增量奖励
        total_reward += rescue_ratio * 10  # 全局奖励
        
        # 时间惩罚
        if self.current_time >= self.max_time:
            total_reward -= 5
        
        # 检查是否结束
        done = self.current_time >= self.max_time or all(task.completed for task in self.tasks)
        
        return self._get_observation(), total_reward, done, {}

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加层归一化
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加层归一化
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class MAPPO:
    def __init__(self, env, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, eps_clip=0.2, 
                 gae_lambda=0.95, entropy_coef=0.01, value_coef=0.5):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # 获取观察和动作空间维度
        obs = env._get_observation()
        self.obs_dim = obs.shape[1]
        self.action_dim = env.num_tasks + 1  # 任务数量 + 不选择动作
        
        # 初始化网络
        self.actor = Actor(self.obs_dim, self.action_dim)
        self.critic = Critic(self.obs_dim)
        self.actor_old = Actor(self.obs_dim, self.action_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # 优化器
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)
        
        # 经验回放缓冲区
        self.memory = []
    
    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs)
            action_probs = self.actor(obs)
            dist = Categorical(action_probs)
            action = dist.sample()
            return action.numpy(), action_probs.numpy()
    
    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def update(self):
        # 将经验转换为张量
        obs = torch.FloatTensor(np.array([t[0] for t in self.memory]))
        actions = torch.LongTensor(np.array([t[1] for t in self.memory]))
        old_probs = torch.FloatTensor(np.array([t[2] for t in self.memory]))
        rewards = torch.FloatTensor(np.array([t[3] for t in self.memory]))
        next_obs = torch.FloatTensor(np.array([t[4] for t in self.memory]))
        dones = torch.FloatTensor(np.array([t[5] for t in self.memory]))
        
        # 计算优势函数
        with torch.no_grad():
            values = self.critic(obs).squeeze(-1)
            next_values = self.critic(next_obs).squeeze(-1)
            advantages = self.compute_gae(rewards, values, next_values, dones)
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 更新策略网络
        for _ in range(10):  # 多次更新
            action_probs = self.actor(obs)
            dist = Categorical(action_probs)
            new_probs = dist.log_prob(actions)
            ratio = torch.exp(new_probs - torch.log(old_probs))
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 添加熵正则化
            entropy = dist.entropy().mean()
            actor_loss = actor_loss - self.entropy_coef * entropy
            
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # 梯度裁剪
            self.optimizer_actor.step()
        
        # 更新价值网络
        for _ in range(10):
            values = self.critic(obs).squeeze(-1)
            critic_loss = F.mse_loss(values, returns)
            
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # 梯度裁剪
            self.optimizer_critic.step()
        
        # 更新旧策略网络
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # 清空经验回放缓冲区
        self.memory = []
    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

def train_mappo(num_episodes=1000, save_interval=100):
    # 创建环境
    env = EmergencyEnv(num_tasks=50, num_groups=5)
    agent = MAPPO(env)
    
    # 创建保存目录
    if not os.path.exists('mappo_models'):
        os.makedirs('mappo_models')
    
    # 训练记录
    episode_rewards = []
    best_reward = float('-inf')
    
    # 训练循环
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # 选择动作
            actions, action_probs = agent.select_action(obs)
            
            # 执行动作
            next_obs, reward, done, _ = env.step(actions)
            
            # 存储经验
            for i in range(len(actions)):
                agent.memory.append((
                    obs[i],
                    actions[i],
                    action_probs[i][actions[i]],
                    reward,
                    next_obs[i],
                    done
                ))
            
            # 更新观察和奖励
            obs = next_obs
            episode_reward += reward
            step_count += 1
            
            # 如果收集了足够的经验，就更新策略
            if len(agent.memory) >= 1000:
                agent.update()
        
        # 记录奖励
        episode_rewards.append(episode_reward)
        
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}, Average Reward (last 10): {avg_reward:.2f}")
            
            # 保存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save('mappo_models/mappo_best.pt')
        
        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            agent.save(f'mappo_models/mappo_episode_{episode + 1}.pt')
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_curve.png')
    plt.close()
    
    return agent

def test_mappo(model_path, num_episodes=10):
    # 创建固定种子的环境
    env = EmergencyEnv(num_tasks=50, num_groups=5, seed=42)
    agent = MAPPO(env)
    
    # 加载模型
    agent.load(model_path)
    
    # 测试循环
    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            actions, _ = agent.select_action(obs)
            
            # 执行动作
            next_obs, reward, done, _ = env.step(actions)
            
            # 更新观察和奖励
            obs = next_obs
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}, Reward: {episode_reward}")
    
    # 打印平均奖励
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")

if __name__ == "__main__":
    # 训练模型
    agent = train_mappo(num_episodes=1000, save_interval=100)
    
    # 测试模型
    test_mappo('mappo_models/mappo_best.pt', num_episodes=10)