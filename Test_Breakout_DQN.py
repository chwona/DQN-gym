################################################################################
# Program : Test_Breakout_DQN.py
# Description : Open AI GYM의 Taxi 환경의 강화학습 구현 코드 (코랩용) : DQN Breakout
################################################################################

##################################################
# install package
##################################################
!pip install -U gym>=0.21.0
!pip install -U gym[atari,accept-rom-license]

## 이미지 캡쳐를 통해 영상으로 확인하려면 주석 해제
# !apt-get install -y xvfb x11-utils
# !pip install pyvirtualdisplay==0.2.* PyOpenGL==3.1.* PyOpenGL-accelerate==3.1.*

##################################################
# import package
##################################################
import gym as gym
import numpy as np
import random
from tensorflow.keras.models import load_model
from skimage import transform, color
from joblib import dump, load

## 이미지 캡쳐를 통해 영상으로 확인하려면 주석 해제
# import matplotlib.pyplot as plt
# from IPython import display as ipythondisplay
# from pyvirtualdisplay import Display

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

##################################################
# Trained Model Load
##################################################
loaded_model = load_model('/content/drive/MyDrive/RL/Breakout/ver3/breakout_model_e409.h5') #학습된 모델 저장경로 설정

## 이미지 캡쳐를 통해 영상으로 확인하려면 주석 해제
# display = Display(visible=0, size=(400, 300))
# display.start()

##################################################
# Set Envrionment
##################################################
env = gym.make('Breakout-v0', render_mode="rgb_array")
num_actions = env.action_space.n
input_shape = env.observation_space.shape
print('state 구조 : {}'.format(input_shape))
print('action 개수 : {}'.format(num_actions))

##################################################
# Test
##################################################
state, _ = env.reset()

## 이미지 캡쳐를 통해 영상으로 확인하려면 주석 해제
# prev_screen = env.render()
# plt.imshow(prev_screen)

test_episodes = []
test_total_rewards = []
test_total_steps = []

# 학습때 진행했던 전처리 진행
action_2 = {0:1, 1:2, 2:3, 3:3}

# Test loop
num_episodes = 10  

for episode in range(num_episodes):
    state, _ = env.reset()

    # 랜덤으로 뽑힌 값 만큼의 프레임동안 움직이지 않음
    for _ in range(random.randint(1, 30)):
        state, _, _, _, _ = env.step(1)

    state = preprocess_frame(state)
    stacked_frame = np.stack([state] * 4, axis=2)

    done = False
    total_reward = 0
    total_step = 0

    while not done:
        # action = env.action_space.sample() # 랜덤 행동
	# 학습된 모델로 예측한 행동 산출
        q_values = loaded_model.predict(np.expand_dims(stacked_frame, axis=0), verbose=0)
        action = np.argmax(q_values)
        
        real_action = action_2[action]

        # 다음 상태, 보상, 종료여부 생성
        next_state, reward, terminated, truncated, info = env.step(real_action)

        ## 이미지 캡쳐를 통해 영상으로 확인하려면 주석 해제
        # screen = env.render()
        # plt.imshow(screen)
        # ipythondisplay.clear_output(wait=True)
        # ipythondisplay.display(plt.gcf())

        next_state = preprocess_frame(next_state)
        next_stacked_frame = np.append(stacked_frame[:, :, 1:], np.expand_dims(next_state, axis=2), axis=2)

        done = (terminated or truncated)
        total_reward += reward
        stacked_frame = next_stacked_frame
        total_step += 1

        if done:
            break

    print(f"Episode {episode + 1}: Total Reward = {total_reward} Total Step = {total_step}")

    test_episodes.append(episode)
    test_total_rewards.append(total_reward)
    test_total_steps.append(total_step)

## 이미지 캡쳐를 통해 영상으로 확인하려면 주석 해제
# ipythondisplay.clear_output(wait=True)
env.close()

# 테스트 결과 값 저장
dump(test_episodes, '/content/drive/MyDrive/RL/breakout_test_episodes.joblib')
dump(test_total_rewards, '/content/drive/MyDrive/RL/breakout_test_total_rewards.joblib')
dump(test_total_steps, '/content/drive/MyDrive/RL/breakout_test_total_steps.joblib')
print("Test Complete!")
