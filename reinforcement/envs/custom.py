import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv
import numpy as np

class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define the angle range (in radians) where termination due to angle is suppressed.
        # Here we assume that if the pole's angle is within a small window around π, we don't want to end the episode.
        self.downward_range = (np.pi - 0.209, np.pi + 0.209)  # ~12° window around π

    def step(self, action):
        # Use the parent class step to get the standard outputs
        obs, reward, terminated, truncated, info = super().step(action)
        theta = obs[2]  # In the state vector, the third element is the pole angle

        # If termination was triggered by the pole angle, but the angle is in the allowed downward range,
        # then override termination.
        if terminated and (self.downward_range[0] <= theta <= self.downward_range[1]):
            terminated = False

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Reset the environment using the parent reset and then modify the initial state.
        # The updated Gymnasium API returns both observation and info.
        observation, info = super().reset(**kwargs)
        # Set the pole angle to π (pointing down) and the angular velocity to 0.
        observation[2] = np.pi
        observation[3] = 0.0
        return observation, info