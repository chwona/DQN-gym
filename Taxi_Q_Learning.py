################################################################################
# Program : Taxi_Q_Learning.py
# Description : Open AI GYM의 Taxi 환경의 강화학습 구현 코드 (코랩용) : Q_Learning Taxi 
################################################################################
##################################################
# install package
##################################################
!apt-get install -y xvfb x11-utils
!pip install pyvirtualdisplay==0.2.* PyOpenGL==3.1.* PyOpenGL-accelerate==3.1.*

##################################################
# import package
##################################################
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load

##################################################
# Define Class and Function 
##################################################
# Q-Table Update
def table_update(state, action, reward, next_state, alpha, gamma):
    best_qa = max([q_table[(next_state, a)] for a in range(env.action_space.n)])
    q_table[(state, action)] += alpha * (reward + gamma * best_qa - q_table[(state, action)])

#엡실론그리디
def epsilon_greedy(state, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: q_table[(state,x)])
		
##################################################
# Hyperparameters
##################################################
alpha = 0.4
gamma = 0.999
epsilon = 0.017

##################################################
# Set environment
##################################################
env = gym.make('Taxi-v3')

# Q-Table 생성
q_table = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q_table[(s,a)] = 0.0

##################################################
# Train
##################################################
num_episodes = 10000
QL_train_Total_Rewards = []

for episode in range(num_episodes):
    total_reward = 0
    state = env.reset()
    
    while True:
        env.render()
        
        # 엡실론 그리디로 행동 선정
        action = epsilon_greedy(state, epsilon)
        
        # 행동에 따른 다음 상태, 보상, 종료여부 정의
        next_state, reward, done, _ = env.step(action)
        
        # Q-Table 업데이트
        table_update(state, action, reward, next_state, alpha, gamma)
        
        state = next_state
        total_reward += reward

        if done:
            break

    QL_train_Total_Rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()

##################################################
# Save
##################################################
# 현재까지 최적화된 Q-Table 저장
dump(q_table, '/content/drive/MyDrive/RL/Taxi_q_table.joblib') 
# 에피소드별 보상 저장
dump(QL_train_Total_Rewards, '/content/drive/MyDrive/RL/Taxi_QL_train_Total_Rewards.joblib')