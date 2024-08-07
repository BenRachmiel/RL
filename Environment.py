import numpy as np


class SimpleEnvironment:
    def __init__(self, speed, angular_velocity, area_size=(10, 10)):
        self.speed = speed
        self.angular_velocity = angular_velocity
        self.area_size = area_size
        self.position = np.zeros(2)
        self.target = np.zeros(2)
        self.orientation = 0.0
        self.done = False
        self._randomize_positions()

    def _randomize_positions(self):
        min_distance = 10.0
        while True:
            self.position = np.random.uniform(0, self.area_size[0], size=2)
            self.target = np.random.uniform(0, self.area_size[1], size=2)
            distance = np.linalg.norm(self.target - self.position)
            if distance >= min_distance:
                break

    def reset(self):
        self._randomize_positions()
        self.orientation = np.random.uniform(0, 2 * np.pi)  # Random orientation
        self.done = False
        return self.get_state()

    def get_state(self):
        return np.concatenate([self.position, [self.orientation], self.target])

    def step(self, action):
        """Execute the action and update the agent's position."""
        # Continuous actions: action is a scalar indicating the angular change
        self.orientation += action * self.angular_velocity

        # Move forward in the direction of the current orientation
        self.position[0] += self.speed * np.cos(self.orientation)
        self.position[1] += self.speed * np.sin(self.orientation)

        # Clip the position to stay within the defined area
        self.position = np.clip(self.position, 0, self.area_size[0])

        # Calculate the distance to the target
        distance_to_target = np.linalg.norm(self.target - self.position)

        # Reward system: Give a reward when reaching the target, and reposition the target
        if distance_to_target < 0.1:
            reward = 100  # Reward for reaching the target
            self.target = np.random.uniform(0, self.area_size[0], size=2)  # New target position
        else:
            reward = -distance_to_target  # Negative reward to minimize distance

        return self.get_state(), reward, self.done
