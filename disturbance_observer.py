# Disturbance estimation by state augmentation.
#
# Extends the PID + Kalman altitude sim: augment the state to x = [z, v, b],
# where b is an unknown CONSTANT vertical disturbance acceleration the plant
# experiences (z_ddot = a_cmd + b_true) but the controller and filter do not
# know. The filter must estimate b from noisy position measurements alone.
import matplotlib
import os
import sys
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Parameters (shared with sim.py) ---
m = 1.0            # mass [kg]
g = 9.81           # gravity [m/s^2]
dt = 0.01          # timestep [s]
steps = 1500       # 15 s
zref = 10.0        # altitude setpoint [m]
sigma_z = 0.25     # position measurement noise std [m]
sigma_a = 1.0      # process noise on [z, v] (unmodeled accel) [m/s^2]
qb = 1e-3          # process noise on the disturbance state b (random walk)
kP, kD = 6.0, 4.0  # the "drooping" PD config from sim.py

# True unknown disturbance the controller/filter are blind to.
# Negative = downward push (e.g. unmodeled payload); worsens the PD droop.
b_true = -2.0      # [m/s^2]

# --- Augmented Kalman model: x = [z, v, b]^T ---
F_aug = np.array([[1.0, dt, 0.5 * dt**2],
                  [0.0, 1.0, dt],
                  [0.0, 0.0, 1.0]])
G = np.array([[0.5 * dt**2], [dt], [0.0]])   # known input a_cmd enters z, v (not b)
H = np.array([[1.0, 0.0, 0.0]])              # measure position only

G2 = np.array([[0.5 * dt**2], [dt]])         # accel -> [z, v] for process noise
Q_aug = np.zeros((3, 3))
Q_aug[:2, :2] = sigma_a**2 * (G2 @ G2.T)     # process noise on z, v
Q_aug[2, 2] = qb                             # random-walk process noise on b
R = np.array([[sigma_z**2]])


def observability(F, Hm):
    """Discrete observability matrix and its rank."""
    n = F.shape[0]
    O = np.vstack([Hm @ np.linalg.matrix_power(F, k) for k in range(n)])
    return O, np.linalg.matrix_rank(O)


def run(feedforward=False, b_of_t=None, seed=0):
    """Simulate one episode. b_of_t(t) overrides the constant b_true if given."""
    rng = np.random.default_rng(seed)
    z, v = 0.0, 0.0
    x_hat = np.array([[0.0], [0.0], [0.0]])   # [z, v, b] estimate
    P = np.eye(3)
    t_hist, z_true, z_est, bhat_hist, btrue_hist = [], [], [], [], []

    for i in range(steps):
        t = i * dt
        bt = b_of_t(t) if b_of_t is not None else b_true

        # controller commands on the ESTIMATED state (PD)
        zhat, vhat, bhat = x_hat[0, 0], x_hat[1, 0], x_hat[2, 0]
        e = zref - zhat
        thrust = kP * e + kD * (-vhat)
        if feedforward:
            thrust += -m * bhat               # cancel the estimated disturbance
        thrust = max(0.0, thrust)
        a_cmd = (thrust - m * g) / m

        # true plant with the unknown disturbance
        acc = a_cmd + bt
        v += acc * dt
        z += v * dt

        # noisy measurement, then augmented Kalman predict + update
        y = z + rng.normal(0.0, sigma_z)
        x_hat = F_aug @ x_hat + G * a_cmd
        P = F_aug @ P @ F_aug.T + Q_aug
        ytilde = y - H @ x_hat
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x_hat = x_hat + K @ ytilde
        P = (np.eye(3) - K @ H) @ P

        t_hist.append(t); z_true.append(z); z_est.append(x_hat[0, 0])
        bhat_hist.append(x_hat[2, 0]); btrue_hist.append(bt)

    return tuple(np.array(a) for a in (t_hist, z_true, z_est, bhat_hist, btrue_hist))


def steady(arr):
    return float(np.mean(arr[-100:]))


if __name__ == "__main__":
    # --- observability of the augmented system from position alone ---
    O, rank = observability(F_aug, H)
    print(f"Augmented observability matrix rank = {rank} / 3  "
          f"(det = {np.linalg.det(O):.3e})  -> "
          f"{'OBSERVABLE' if rank == 3 else 'NOT observable'}")
    print(f"b_true = {b_true:+.2f} m/s^2\n")

    # --- Case 1: disturbance present, NO feedforward ---
    t, z1, ze1, bh1, bt1 = run(feedforward=False)
    print("Case 1 (disturbance, no feedforward):")
    print(f"  final b_hat        = {bh1[-1]:+.3f} m/s^2  (true {b_true:+.2f})")
    print(f"  steady-state alt   = {steady(z1):.3f} m")

    # --- Case 2: same, WITH estimated-disturbance feedforward ---
    t, z2, ze2, bh2, bt2 = run(feedforward=True)
    print("\nCase 2 (disturbance, feedforward -m*b_hat):")
    print(f"  final b_hat        = {bh2[-1]:+.3f} m/s^2")
    print(f"  steady-state alt   = {steady(z2):.3f} m")

    # --- reference droop with NO disturbance (b_true = 0) ---
    _b = b_true
    b_true = 0.0
    t, z0, *_ = run(feedforward=False)
    b_true = _b
    droop0 = steady(z0)
    print(f"\nReference (no disturbance) steady-state alt = {droop0:.3f} m")

    # --- Case 3: time-varying disturbance (constant-b model must lag) ---
    b_of_t = lambda tt: 2.0 * np.sin(2 * np.pi * 0.1 * tt)   # 0.1 Hz, +/-2 m/s^2
    t, z3, ze3, bh3, bt3 = run(feedforward=False, b_of_t=b_of_t)
    lag_rmse = float(np.sqrt(np.mean((bh3 - bt3) ** 2)))
    print(f"\nCase 3 (time-varying b): tracking RMSE = {lag_rmse:.3f} m/s^2")

    # --- figure ---
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    ax[0, 0].plot(t, z1, label="true altitude", color="tab:blue")
    ax[0, 0].plot(t, ze1, "--", label="estimated altitude", color="tab:cyan")
    ax[0, 0].axhline(zref, color="k", ls=":", lw=1, label="setpoint (10 m)")
    ax[0, 0].axhline(droop0, color="tab:gray", ls="--", lw=1, label=f"no-disturbance droop ({droop0:.2f} m)")
    ax[0, 0].set_title("Case 1: disturbance present, no feedforward")
    ax[0, 0].set_xlabel("time (s)"); ax[0, 0].set_ylabel("altitude (m)")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True)

    ax[0, 1].plot(t, bt1, label="true b", color="k", lw=2)
    ax[0, 1].plot(t, bh1, label="estimated b̂", color="tab:red")
    ax[0, 1].set_title("Case 1: disturbance estimate converges")
    ax[0, 1].set_xlabel("time (s)"); ax[0, 1].set_ylabel("b (m/s²)")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True)

    ax[1, 0].plot(t, z1, label=f"no feedforward ({steady(z1):.2f} m)", color="tab:red")
    ax[1, 0].plot(t, z2, label=f"with feedforward ({steady(z2):.2f} m)", color="tab:green")
    ax[1, 0].axhline(zref, color="k", ls=":", lw=1, label="setpoint (10 m)")
    ax[1, 0].axhline(droop0, color="tab:gray", ls="--", lw=1, label=f"no-disturbance droop ({droop0:.2f} m)")
    ax[1, 0].set_title("Case 2: feedforward cancels the disturbance (not the gravity droop)")
    ax[1, 0].set_xlabel("time (s)"); ax[1, 0].set_ylabel("altitude (m)")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True)

    ax[1, 1].plot(t, bt3, label="true b(t)", color="k", lw=2)
    ax[1, 1].plot(t, bh3, label="estimated b̂", color="tab:red")
    ax[1, 1].set_title("Case 3: time-varying disturbance — constant-b model lags")
    ax[1, 1].set_xlabel("time (s)"); ax[1, 1].set_ylabel("b (m/s²)")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(True)

    fig.suptitle("Disturbance estimation by state augmentation (x = [z, v, b])", fontsize=13)
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "disturbance_observer.png"), dpi=150)
    if "--no-show" not in sys.argv:
        plt.show()
    plt.close()
