"""Benchmark the hand-tuned PID against the trained PPO policy.

Runs both controllers through identical episodes on the same environment,
reports steady-state error / overshoot / settling time, and produces the
comparison figure. Also runs a generalization test: PPO was trained only at
z_target = 10 m, so we evaluate both controllers at unseen setpoints.

    .venv/Scripts/python evaluate.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs.altitude_env import AltitudeEnv
from controllers.pid import PIDPolicy

# Classical baseline re-tuned for the normalized/hover-offset action space.
# Integral is 0: with gravity fed forward there is no offset to integrate away.
PID_GAINS = dict(kp=1.5, ki=0.0, kd=0.8)

TRAIN_SETPOINT = 10.0                 # PPO was trained here
SETPOINTS = [10.0, 6.0, 14.0]         # 10 = trained, others = generalization
N_SEEDS = 10                          # metrics averaged over seeds (env noise varies)


def run_episode(env, controller, is_rl=False, seed=0):
    obs, _ = env.reset(seed=seed)
    if hasattr(controller, "reset"):
        controller.reset()
    ts, zs, t = [], [], 0.0
    done = False
    while not done:
        if is_rl:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            action = controller.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        zs.append(env.z)              # TRUE altitude (filter sits upstream of control)
        ts.append(t); t += env.dt
        done = terminated or truncated
    return np.array(ts), np.array(zs)


def metrics(ts, zs, setpoint, band=0.02):
    y_ss = float(np.mean(zs[-50:]))
    e_ss = abs(setpoint - y_ss)
    overshoot = max(0.0, (float(np.max(zs)) - setpoint) / setpoint * 100.0)
    tol = band * setpoint
    outside = np.where(np.abs(zs - setpoint) > tol)[0]
    settling_time = float(ts[outside[-1]]) if len(outside) else 0.0
    return dict(steady_state_error=e_ss, overshoot_pct=overshoot,
                settling_time=settling_time)


def avg_metrics(make_controller, setpoint, is_rl):
    """Average metrics over N_SEEDS episodes at a given setpoint."""
    acc = {"steady_state_error": [], "overshoot_pct": [], "settling_time": []}
    for s in range(N_SEEDS):
        env = AltitudeEnv(z_target=setpoint)
        ts, zs = run_episode(env, make_controller(env), is_rl=is_rl, seed=s)
        m = metrics(ts, zs, setpoint)
        for k in acc:
            acc[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    model = PPO.load("ppo_altitude")

    def make_pid(env):
        return PIDPolicy(dt=env.dt, **PID_GAINS)

    def make_ppo(env):
        return model

    # ---- metrics table ----
    print(f"{'setpoint':>9} {'controller':>10} | {'e_ss':>6} {'overshoot%':>10} {'settle_s':>8}")
    for sp in SETPOINTS:
        tag = "(train)" if sp == TRAIN_SETPOINT else "(unseen)"
        for name, is_rl, maker in [("PID", False, make_pid), ("PPO", True, make_ppo)]:
            m = avg_metrics(maker, sp, is_rl)
            print(f"{sp:6.1f} {tag} {name:>10} | {m['steady_state_error']:6.3f} "
                  f"{m['overshoot_pct']:10.1f} {m['settling_time']:8.2f}")
        print()

    # ---- comparison figure: one panel per setpoint, both controllers overlaid ----
    fig, axes = plt.subplots(1, len(SETPOINTS), figsize=(15, 4.5), sharey=False)
    for ax, sp in zip(axes, SETPOINTS):
        env = AltitudeEnv(z_target=sp)
        t_pid, z_pid = run_episode(env, make_pid(env), is_rl=False, seed=0)
        env = AltitudeEnv(z_target=sp)
        t_rl, z_rl = run_episode(env, model, is_rl=True, seed=0)
        ax.plot(t_pid, z_pid, label="PID", color="tab:blue", linewidth=1.8)
        ax.plot(t_rl, z_rl, label="PPO", color="tab:orange", linewidth=1.8)
        ax.axhline(sp, linestyle="--", color="k", linewidth=1, label="setpoint")
        tag = "trained" if sp == TRAIN_SETPOINT else "generalization (unseen)"
        ax.set_title(f"{sp:.0f} m target ({tag})")
        ax.set_xlabel("time (s)")
        ax.grid(True)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("altitude (m)")
    fig.suptitle("PID vs PPO on 1-DOF altitude control (Kalman-filtered observation)", fontsize=13)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/comparison.png", dpi=150)
    print("saved results/comparison.png")


if __name__ == "__main__":
    main()
