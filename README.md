# 1D Quadrotor Altitude: PID + Kalman Filter

![comparison](comparison.png)

Vertical altitude control for a quadrotor, in Python.

A PID's derivative term differentiates whatever you feed it, and differentiating a noisy signal makes the noise worse. To fix this, I added a Kalman filter to sidestep it by estimating velocity as part of the state instead of differencing the position signal.

## What happens

P-only oscillates forever. The acceleration is proportional to displacement, and there's nothing proportional to velocity to dampen energy. This leaves us with the equation for shm, which oscillates indefinitely.

PD stops oscillating but settles low, around 8.37 m. The derivative term adds the missing damping, so the oscillation stops. However, the P term only pushes in proportion to how far off target the drone is. To generate enough thrust to cancel gravity, it needs some error to push against, so it settles at a fixed distance below the target.

PID hits the target. The integral term accumulates error over time, so its output keeps growing as long as any error remains, unlike P and D, which collapse to zero when the error does. The system can only settle when the error is zero and when the integral's thrust cancels gravity.

## What the Kalman filter does

The controller only sees a noisy sensor reading, which it feeds to the derivative term, which then exaggerates the noise. Adding a filter sits between the sensor and the controller and provides a clean estimate of altitude and velocity instead.

At each step, it predicts where the drone should be using the physics model, then corrects that guess against the new reading. K, the Kalman gain, sets how far to trust the sensor over the model, weighted by their relative uncertainty. A noisy sensor pulls K down and leans on the model, an unreliable model pushes it up, and it rebalances every step.

As shown in the plot, the filter estimates velocity as part of the state rather than differentiating noisy signals. This means the controller can use the filter's velocity estimate directly, rather than differentiating a noisy signal, and that the estimate tracks the true state.

**Note:** The Kalman predict step feeds it the commanded acceleration. I started with a=0, but it blew up in every configuration. P-only got pumped past 130 m, PD had a huge positive bias. This was because, with a=0, the drone can't be seen accelerating, so the estimate lags the truth, creating a phase lag, and destabilizing the feedback loop.

## Run it

    pip install numpy matplotlib
    python sim.py

This writes `comparison.png`, the figure shown above.

## Files

- `sim.py` —
