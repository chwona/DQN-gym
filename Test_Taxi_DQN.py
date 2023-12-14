################################################################################
# Program : Test_Taxi_DQN.py
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
from tensorflow.keras.models import load_model

## 이미지 캡쳐를 통해 영상으로 확인하려면 주석 해제
# import matplotlib.pyplot as plt
# from IPython import display as ipythondisplay
# from pyvirtualdisplay import Display

##################################################
# Trained Model Load
##################################################
# Load the saved model
loaded_model = load_model('/content/drive/MyDrive/RL/taxi_model.h5') #모델이 저장된 경로 설정

## 이미지 캡쳐를 통해 영상으로 확인하려면 주석 해제
# display = Display(visible=0, size=(400, 300))
# display.start()

##################################################
# Set Envrionment
##################################################
env = gym.make("Taxi-v3")
state = env.reset()
state = np.reshape(state, [1, 1])

## 이미지 캡쳐를 통해 영상으로 확인하려면 주석 해제
# prev_screen = env.render(mode='rgb_array')
# plt.imshow(prev_screen)

##################################################
# Test
##################################################
test_episodes = []
test_total_rewards = []
test_total_steps = []

num_episodes = 100

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, 1])
    done = False
    total_reward = 0
    total_step = 0
	
    while not done:
        # action = env.action_space.sample()
		# 학습된 모델로 예측한 행동 산출
        action = np.argmax(loaded_model.predict(state, verbose=0))
		# 다음 상태, 보상, 종료여부 생성
        next_state, reward, done, info = env.step(action)
        
        ## 이미지 캡쳐를 통해 영상으로 확인하려면 주석 해제
        # screen = env.render(mode='rgb_array')
        # plt.imshow(screen)
        # ipythondisplay.clear_output(wait=True)
        # ipythondisplay.display(plt.gcf())

        state = next_state
        state = np.reshape(next_state, [1, 1])
        total_reward += reward
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
dump(test_episodes, '/content/drive/MyDrive/RL/taxi_test_episodes.joblib')
dump(test_total_rewards, '/content/drive/MyDrive/RL/taxi_test_total_rewards.joblib')
dump(test_total_steps, '/content/drive/MyDrive/RL/taxi_test_total_steps.joblib')
print("Test Complete!")