################################################################################
# Program : Train_Breakout_DQN.py
# Description : Open AI GYM의 Taxi 환경의 강화학습 구현 코드 (코랩용) : DQN Breakout
################################################################################

##################################################
# install package
##################################################
!pip install -U gym>=0.21.0
!pip install -U gym[atari,accept-rom-license]

##################################################
# import package
##################################################
import gym
import numpy as np
from skimage import transform, color

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

from collections import deque
import random

from joblib import dump, load

##################################################
# Define Class and Function
##################################################
#전처리용
def preprocess_frame(frame):
    cropped_frame = frame[35:195, 8:152]
    grayscale_frame = color.rgb2gray(cropped_frame)
    normalized_frame = grayscale_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, (84, 84))
    preprocessed_frame = preprocessed_frame.astype(np.uint8)

    return preprocessed_frame

#엡실론그리디
class EpsilonGreedy:
    def __init__(self, max_epsilon, min_epsilon, dacay_rate):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.dacay_rate = dacay_rate
        self.epsilon = max_epsilon

    def exploration_rate(self, step):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.dacay_rate * step)
        return self.epsilon

# DNN 모델 생성
def build_model(input_shape, num_actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    return model

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add_exp(self, stacked_frame, action, reward, next_stacked_frame, done):
        self.buffer.append((stacked_frame, action, reward, next_stacked_frame, done))

    def sample_mbatch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        stacked_frames, actions, rewards, next_stacked_frames, dones = zip(*batch)
        return np.array(stacked_frames), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_stacked_frames), np.array(dones, dtype=np.uint8)
        # return np.stack(stacked_frames), np.stack(actions), np.stack(rewards, dtype=np.float32), np.stack(next_stacked_frames), np.stack(dones, dtype=np.uint8)

##################################################
# Hyperparameters
##################################################
max_mermory_size = 100000000
batch_size = 32
gamma = 0.99  # Discount factor
max_epsilon = 1.0
min_epsilon = 0.01
dacay_rate = 0.001
#dacay_rate = 0.0001
update_period = 10 #에피소드 기준

##################################################
# Set Envrionment
##################################################
env = gym.make('Breakout-v0')
num_actions = env.action_space.n
input_shape = env.observation_space.shape
print('state 구조 : {}'.format(input_shape))
print('action 개수 : {}'.format(num_actions))

# 전처리한 버전으로 shape 재 설정
input_shape = (84, 84, 4)

# 모델 생성
dqn = build_model(input_shape, num_actions)
target_dqn = build_model(input_shape, num_actions)
target_dqn.set_weights(dqn.get_weights())

# Replay Buffer 설정
replay_buffer = ReplayBuffer(max_mermory_size)

# 엡실론그리디
exploration_strategy = EpsilonGreedy(max_epsilon, min_epsilon, dacay_rate)

##################################################
# Train
##################################################
train_episodes = []
episode_rewards = [] # train_total_rewards
train_total_steps = []
train_epsilon = []

num_episodes = 5000
no_op_steps = 30

for episode in range(num_episodes):

    state, _ = env.reset()
    # 초반 30 프레임 제외
    for _ in range(random.randint(1, no_op_steps)):
        state, _, _, _, _ = env.step(1)

    # 프레임을 전처리 이후 가장 첫번째 stacked_frame은 같은 state 4개 stack해서 사용
    state = preprocess_frame(state)
    stacked_frame = np.stack([state] * 4, axis=2)

    # reset
    episode_reward = 0
    total_step = 0
    done = False

    # action 0 (NOOP) -> Fire 대체하여 게임 재시작하도록 유도
    action_set = {0:1, 1:2, 2:3, 3:3}

    while not done:
        # rate = exploration_strategy.exploration_rate(total_step)
        rate = exploration_strategy.exploration_rate(episode)
        if np.random.rand() < rate:
            action = env.action_space.sample()
        else:
            # q_values = dqn.predict(np.expand_dims(state, axis=0))
            q_values = dqn.predict(np.expand_dims(stacked_frame, axis=0), verbose=0) #stacked_frames
            action = np.argmax(q_values)

        # action 0 (NOOP) -> Fire 대체하여 게임 재시작하도록 유도
        action_2 = action_set[action]

        next_state, reward, terminated, truncated, _ = env.step(action_2)
        # 다음 상태 전처리
        next_state = preprocess_frame(next_state)
        # stacked_frame에 새로 쌓기
        next_stacked_frame = np.append(stacked_frame[:, :, 1:], np.expand_dims(next_state, axis=2), axis=2)

        done = (terminated or truncated)
        episode_reward += reward

        # 경험 저장
        replay_buffer.add_exp(stacked_frame, action, reward, next_stacked_frame, done)

        stacked_frame = next_stacked_frame
        total_step += 1

        if done:
            break

    # Mini-Batch 학습
    if len(replay_buffer.buffer) > 3000: # 경험이 3000개 초과 저장되면
    # if len(replay_buffer.buffer) > batch_size:

        # 32 * 10 = 320개 Row 학습
        for i in range(10):

            stacked_frames, actions, rewards, next_stacked_frames, dones = replay_buffer.sample_mbatch(batch_size)

            # 일반 네트워크에서의 예측값
            targets = dqn.predict(stacked_frames, verbose=0)

            # s'로 인한 타겟 네트워크에서의 예측값
            next_q = target_dqn.predict(next_stacked_frames, verbose=0)

            # 벨먼 방정식 적용 후 학습
            targets[np.arange(batch_size), actions] = rewards + gamma * np.max(next_q, axis=1) * (1 - dones)
            dqn.train_on_batch(stacked_frames, targets)

		# 주기별 일반네트워크의 가중치 타겟 네트워크로 적용
        if episode % update_period == 0 and episode != 0:
            target_dqn.set_weights(dqn.get_weights())

    train_episodes.append(episode)
    episode_rewards.append(episode_reward)
    train_total_steps.append(total_step)
    train_epsilon.append(rate)

    # 주기별 모델 저장
    if (episode + 1) % 100 == 0:
        target_dqn.save(f'/content/drive/MyDrive/RL/Breakout/breakout_model_e{episode}.h5')

    # 중간에 끊겼을 때 저장용
    dump(episode, '/content/drive/MyDrive/RL/Breakout/breakout_latest_episode.joblib')
    dump(episode_reward, '/content/drive/MyDrive/RL/Breakout/breakout_latest_episode_reward.joblib')
    dump(rate, '/content/drive/MyDrive/RL/Breakout/breakout_latest_exploration_rate.joblib')

    print(f"Episode: {episode + 1}/{num_episodes}, Total Reward: {episode_reward}, Epsilon: {rate}, Total step: {total_step}")

#마지막 에피소드의 모델 저장
target_dqn.save(f'/content/drive/MyDrive/RL/Breakout/breakout_model_e{episode}.h5')
dump(train_episodes, '/content/drive/MyDrive/RL/Breakout/breakout_train_episodes.joblib')
dump(episode_rewards, '/content/drive/MyDrive/RL/Breakout/breakout_episode_rewards.joblib')
dump(train_total_steps, '/content/drive/MyDrive/RL/Breakout/breakout_train_total_steps.joblib')
dump(train_epsilon, '/content/drive/MyDrive/RL/Breakout/breakout_train_epsilon.joblib')

print("Train Complete!")
