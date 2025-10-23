# 月球登录器强化学习模型训练

import gymnasium as gym
from stable_baselines3 import A2C,PPO
env = gym.make('LunarLander-v3')
env.reset()
# 查看环境的动作空间
# print('sample action',env.action_space.sample())
# # 查看环境的观测空间的形状
# print('observation space shape',env.observation_space.shape)
# # 查看环境的观测空间
# print('sample observation', env.observation_space.sample())
#
# for step in range(200):
#     # 调用 env.render() 时会弹出一个窗口，显示当前环境的状态（例如月球着陆器的位置、速度等）。
#     env.render()
#     # 随机选择一个动作并执行，然后获取环境的反馈。
#     # observation: 执行动作后的新观测值（状态）。
#     # reward: 执行动作后获得的奖励。
#     # terminated: 是否达到终止状态（例如任务完成或失败）。
#     # truncated: 是否因为步数限制而被截断（例如超过最大步数）。
#     # info: 额外的调试信息（通常是一个字典）。
#     observation,reward,terminated,truncated,info=env.step(env.action_space.sample())
#     # 打印当前步的奖励和终止状态。
#     print(reward,terminated)

# 使用A2C或PPO算法


env=gym.make('LunarLander-v3',render_mode='human')
env.reset()
#model = A2C('MlpPolicy', env, verbose=1)
model = PPO('MlpPolicy', env, verbose=1)
#创建了一个 A2C 模型实例，准备用于训练。
model.learn(total_timesteps=100)
#模型会在环境中进行 100 步的交互，并根据 A2C 算法更新策略。
episodes=100
#回合数
vec_env=model.get_env()
#模型训练时使用的环境实例。
obs = vec_env.reset()
#obs 是环境的初始观测值。
for ep in range(episodes):
    #开始测试循环，共运行 episodes 个回合。
    done = False
    while not done:
        action, _states = model.predict(obs)
        #使用训练好的模型预测动作。
        obs, rewards, done, info = vec_env.step(action)
        #在环境中执行动作，并获取反馈。
        env.render()
        #渲染环境，显示当前状态。
        print(rewards)
env.close()
