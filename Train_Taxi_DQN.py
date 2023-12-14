################################################################################
# Program : Train_Taxi_DQN.py
# Description : Open AI GYM의 Taxi 환경의 강화학습 구현 코드 (코랩용) : DQN Taxi
################################################################################

##################################################
# install package
##################################################
!apt-get install -y xvfb x11-utils
!pip install pyvirtualdisplay==0.2.* PyOpenGL==3.1.* PyOpenGL-accelerate==3.1.*

##################################################
# import package
##################################################
import gym
import numpy as np
import random

from keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from keras.optimizers import Adam
from collections import deque
from joblib import dump, load

##################################################
# Define Class and Function
##################################################
# 엡실론그리디 함수
class EpsilonGreedy:
    def __init__(self, max_epsilon, min_epsilon, decay_rate):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.epsilon = max_epsilon

    def exploration_rate(self, step): # 기준값
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * step)
        return self.epsilon

# DNN 모델 생성
def build_model(input_shape, num_actions):
    model = Sequential()
    model.add(Embedding(input_shape, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    return model

##################################################
# Hyperparameters
##################################################
# 엡실론그리디
MAX_EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY_RATE = 0.001

# 일반 네트워크의 가중치 -> 타겟 네트워크 복제 주기 정의
update_period = 10 # 에피소드 기준
gamma = 0.99  # Discount factor
batch_size = 32
memory = deque(maxlen=100000000) # 최대 저장 공간 for Replay Buffer

##################################################
# Set Envrionment
##################################################
# 환경 정의
# env = gym.make('Taxi-v3').env # time limit 제거 버전
env = gym.make('Taxi-v3')

# state , action 개수 정의
num_actions = env.action_space.n
input_shape = env.observation_space.n
print('state 개수 : {}'.format(input_shape))
print('action 개수 : {}'.format(num_actions))

# 모델 생성
dqn = build_model(input_shape, num_actions)
target_dqn = build_model(input_shape, num_actions)
target_dqn.set_weights(dqn.get_weights())
dqn.summary()

#엡실론그리디
exploration_strategy = EpsilonGreedy(MAX_EPSILON, MIN_EPSILON, DECAY_RATE)

##################################################
# Train
##################################################
train_episodes = []
episode_rewards = [] # train_total_rewards
train_total_steps = []
train_epsilon = []

num_episodes = 1000

for episode in range(num_episodes):

    state = env.reset()
    state = np.reshape(state, [1, 1])
	
    episode_reward = 0
    total_step = 0
    done = False

    while not done: # 종료여부 = False
        # action = epsilon_greedy_action(state)

        # 엡실론그리디
        # rate = exploration_strategy.exploration_rate(total_step)
        rate = exploration_strategy.exploration_rate(episode)

        # 기준치보다 낮으면 랜덤 행동 (엡실론그리디)
        if np.random.rand() < rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(dqn.predict(state, verbose=0))

        # 다음 상태, 보상, 종료여부 정의
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 1])

        # 경험 저장 (Replay Buffer)
        memory.append((state, action, reward, next_state, done))

        state = next_state
        episode_reward += reward
        # print(f'episode_reward: {episode_reward}, memory: {len(memory)}')
        total_step += 1

        if done:
            break

    # Mini-Batch 학습
    if len(memory) > 3000: # 경험이 3000개 초과 저장되면
    # if len(memory) > batch_size:
        # print(f'train 시작')

        # 32 * 50 = 1600개 Row 학습
        for i in range(50):
            minibatch = random.sample(memory, batch_size)

            for state_m, action_m, reward_m, next_state_m, done_m in minibatch:
                # 일반 네트워크에서의 예측값
                targets = dqn.predict(state_m, verbose=0)

                if not done_m:
                    # s'로 인한 타겟 네트워크에서의 예측값
                    next_q = target_dqn.predict(next_state_m, verbose=0)
                    # 벨먼 방정식 적용
                    targets[0][action] = reward_m + gamma * np.amax(next_q)
                else:
                    targets[0][action] = reward_m
                # 학습
                dqn.fit(state_m, targets, epochs=1, verbose=0)

		# 주기별 일반네트워크의 가중치 타겟 네트워크로 적용
        if episode % update_period == 0 and episode != 0:
            target_dqn.set_weights(dqn.get_weights())

    train_episodes.append(episode)
    episode_rewards.append(episode_reward)
    train_total_steps.append(total_step)
    train_epsilon.append(rate)

    # 주기별로 모델 저장
    if (episode + 1) % 200 == 0:
        target_dqn.save(f'/content/drive/MyDrive/RL/Taxi/taxi_model_e{episode}.h5')

    # 중간에 끊겼을 때 저장용
    dump(episode, '/content/drive/MyDrive/RL/Taxi/taxi_latest_episode.joblib')
    dump(episode_reward, '/content/drive/MyDrive/RL/Taxi/taxi_latest_episode_reward.joblib')
    dump(rate, '/content/drive/MyDrive/RL/Taxi/taxi_latest_exploration_rate.joblib')

    print(f"Episode: {episode + 1}/{num_episodes}, Memory: {len(memory)}, Total Reward: {episode_reward}, Epsilon: {rate}, Total step: {total_step}")

# 마지막 에피소드의 모델 저장
target_dqn.save(f'/content/drive/MyDrive/RL/Taxi/taxi_model_e{episode}.h5')
dump(train_episodes, '/content/drive/MyDrive/RL/Taxi/taxi_train_episodes.joblib')
dump(episode_rewards, '/content/drive/MyDrive/RL/Taxi/taxi_episode_rewards.joblib')
dump(train_total_steps, '/content/drive/MyDrive/RL/Taxi/taxi_train_total_steps.joblib')
dump(train_epsilon, '/content/drive/MyDrive/RL/Taxi/taxi_train_epsilon.joblib')

print("Train Complete!")
