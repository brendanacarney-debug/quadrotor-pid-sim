import numpy as np


class PIDPolicy:
    """PID control law packaged with the same act(obs) interface as the RL policy.

    obs = [altitude error, vertical velocity], both Kalman-filtered. The
    derivative term uses the velocity estimate directly (d/dt of a constant
    setpoint minus altitude is -velocity), matching the Kalman refactor in
    sim.py rather than differencing a noisy error signal.

    The action is normalized thrust in [-1, 1] where 0 = hover, so gravity is
    already fed forward; the gains only need to command deviations from hover.
    """

    def __init__(self, kp, ki, kd, dt):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.reset()

    def reset(self):
        self.integral = 0.0

    def act(self, obs):
        e = float(obs[0])          # altitude error (target - z_hat)
        v = float(obs[1])          # vertical velocity estimate
        self.integral += e * self.dt
        deriv = -v                 # d/dt(error) = -velocity (setpoint constant)
        u = self.kp * e + self.ki * self.integral + self.kd * deriv
        return np.array([np.clip(u, -1.0, 1.0)], dtype=np.float32)
