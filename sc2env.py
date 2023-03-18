# %%
import gym
from gym import spaces
import numpy as np
import subprocess
import pickle
import time
import os

class Sc2Env(gym.Env):
	def __init__(self):
		super(Sc2Env, self).__init__()
		self.action_space = spaces.Discrete(6)
		self.observation_space = spaces.Box(low=0, high=255,
											shape=(224, 224, 3), dtype=np.uint8)

	def step(self, action):
		wait_for_action = True
		while wait_for_action:
			try:
				with open('state_rwd_action.pkl', 'rb') as f:
					state_rwd_action = pickle.load(f)

					if state_rwd_action['action'] is not None:
						wait_for_action = True
					else:
						wait_for_action = False
						state_rwd_action['action'] = action
						with open('state_rwd_action.pkl', 'wb') as f:
							pickle.dump(state_rwd_action, f)
			except Exception as e:
				pass

		wait_for_state = True
		while wait_for_state:
			try:
				if os.path.getsize('state_rwd_action.pkl') > 0:
					with open('state_rwd_action.pkl', 'rb') as f:
						state_rwd_action = pickle.load(f)
						if state_rwd_action['action'] is None:
							wait_for_state = True
						else:
							state = state_rwd_action['state']
							reward = state_rwd_action['reward']
							done = state_rwd_action['done']
							wait_for_state = False

			except Exception as e:
				wait_for_state = True   
				map = np.zeros((224, 224, 3), dtype=np.uint8)
				observation = map
				data = {"state": map, "reward": 0, "action": 3, "done": False}  
				with open('state_rwd_action.pkl', 'wb') as f:
					pickle.dump(data, f)

				state = map
				reward = 0
				done = False
				action = 3

		info ={}
		observation = state
		return observation, reward, done, info


	def reset(self):
		print("RESETTING ENVIRONMENT")
		map = np.zeros((224, 224, 3), dtype=np.uint8)
		observation = map
		data = {"state": map, "reward": 0, "action": None, "done": False}
		with open('state_rwd_action.pkl', 'wb') as f:
			pickle.dump(data, f)

	
		subprocess.Popen(['python', 'sc2ai.py'])
		return observation  


