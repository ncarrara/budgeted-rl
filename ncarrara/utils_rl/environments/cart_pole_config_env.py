import numpy as np
import math
import gym.envs.classic_control as cc
import gym.spaces as spaces


class CartPoleConfigEnv(cc.CartPoleEnv):

    ID="CartPoleConfig-v0"

    def __init__(self, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5,
                 force_mag=10.0, tau=0.02, x_threshold=2.4,
                 theta_threshold_radians=12 * 2 * math.pi / 360,
                 kinematics_integrator='euler'):
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = (self.masspole + self.masscart)
        self.length = length  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = force_mag
        self.tau = tau  # seconds between state updates
        self.kinematics_integrator = kinematics_integrator

        # Angle at which to fail the episode
        self.theta_threshold_radians = theta_threshold_radians
        self.x_threshold = x_threshold

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
