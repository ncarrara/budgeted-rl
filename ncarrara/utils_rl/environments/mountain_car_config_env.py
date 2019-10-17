import numpy as np
import math
import gym.envs.classic_control as cc
import gym.spaces as spaces


class MountainCarConfigEnv(cc.MountainCarEnv):
    ID = "MountainCarConfig-v0"

    def __init__(self, min_position=-1.2, max_position=0.6, max_speed=0.07, goal_position=0.5):
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high)

        self.seed()
        self.reset()
