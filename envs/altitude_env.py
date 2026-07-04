import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AltitudeEnv(gym.Env):
    """1-DOF vertical altitude control, wrapped as a Gymnasium environment.

    The plant dynamics are copied verbatim from sim.py (semi-implicit Euler).
    A noisy altitude measurement is filtered by a 2-state (z, v) Kalman filter,
    and the *filtered estimate* is what the controller observes. Both the PID
    and the PPO policy therefore act on the identical clean estimate -- the KF
    is a shared perception front-end, not part of the contest.

    Reward and termination are computed on the TRUE state (privileged during
    training); the controller never sees it.
    """

    metadata = {"render_modes": []}

    def __init__(self, dt=0.01, max_steps=1000, z_target=10.0,
                 sigma_z=0.25, sigma_a=1.0):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.z_target = z_target
        self.m, self.g = 1.0, 9.81             # mass, gravity (from sim.py)
        self.T_max = 2.0 * self.m * self.g      # realistic thrust-to-weight = 2
        self.sigma_z = sigma_z                  # measurement noise std (m)
        self.sigma_a = sigma_a                  # process noise std (m/s^2)

        # Kalman model matrices (state x = [z, v]^T), from sim.py
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.G = np.array([[0.5 * dt**2], [dt]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = sigma_a**2 * (self.G @ self.G.T)
        self.R = np.array([[sigma_z**2]])

        # action: normalized thrust command in [-1, 1] (0 = hover)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        # observation: filtered [altitude error, vertical velocity]
        high = np.array([np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _obs(self):
        # filtered estimate: error = target - z_hat, velocity = v_hat
        return np.array([self.z_target - self.x_hat[0, 0], self.x_hat[1, 0]],
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.z, self.vz, self.steps = 0.0, 0.0, 0
        self.x_hat = np.array([[0.0], [0.0]])   # filter estimate
        self.P = np.eye(2)                       # filter covariance
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(action, -1.0, 1.0)[0])
        hover = self.m * self.g
        T = hover + a * (self.T_max - hover)     # map action -> thrust
        az = (T - self.m * self.g) / self.m      # dynamics (from sim.py)
        self.vz += az * self.dt
        self.z += self.vz * self.dt
        self.steps += 1

        # noisy measurement, then Kalman predict (with commanded accel) + update
        y = self.z + self.np_random.normal(0.0, self.sigma_z)
        self.x_hat = self.F @ self.x_hat + self.G * az
        self.P = self.F @ self.P @ self.F.T + self.Q
        ytilde = y - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ ytilde
        self.P = (np.eye(2) - K @ self.H) @ self.P

        # reward/termination on TRUE state; error normalized for sane scale
        e_true = self.z_target - self.z
        reward = -((e_true / self.z_target) ** 2) - 0.01 * (a ** 2)
        terminated = bool(abs(self.z) > 3.0 * self.z_target)   # left the world
        truncated = bool(self.steps >= self.max_steps)
        return self._obs(), reward, terminated, truncated, {}
