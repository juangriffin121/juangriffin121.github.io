---
layout: ../../layouts/MarkdownLayout.astro
title: 'Inverted pendulum'
description: 'Inverted pendulum control system modeling and simulation'
tags: ["Python", "C", "Computational simulation"]
---

*Code: [Data-Model-PID-Sim](https://github.com/juangriffin121/Data-Model-PID-Sim)*

<div style="display: flex; gap: 10px; align-items: center;">
    <video src="/inverted-pendulum/real_system.mp4" controls width="50%" height="50%">
    </video>
    <video src="/inverted-pendulum/full_simulation.mp4" controls width="50%">
    </video>
</div>

<!--![Placeholder: hero shot of the physical setup](./images/hero.png)
<!-- TODO: add a photo/gif of the printer-carriage pendulum rig -->

## Introduction

This project builds and simulates a **control system for an inverted pendulum**, where the pivot point is moved by the carriage mechanism ("the car") salvaged from an old paper printer — the same lead-screw and stepper-motor assembly that normally drags a print head across a page. An Arduino reads the pendulum's angle from a 3D-printed rotary encoder and commands the stepper motor to shuttle the carriage back and forth, using the resulting inertial kick to keep the pendulum balanced on end.

This project is the software part of a joint project co-started with a friend who built the physical rig; we both worked on both sides, but I focused mainly on the simulation and he on the hardware. 

## Motivation

Working on hardware has many complexities that make experimentation hard, this is why I decided to build a simulation of the system to find strategies that could work on the real system, diagnose problems we may encounter and how to deal with them. 

Tuning a PID controller by trial and error directly on hardware is slow and occasionally destructive — bad gains mean the pendulum (and sometimes the printer carriage) crashes into its mechanical limits. A simulation allows for:

- Experimenting with different control systems and sweep PID gains cheaply and see the response before risking hardware.
- Diagnosing problems and limitations of the hardware.
- Derive theoretical limits (e.g. "the PID system can never recover from more than ~6° of deviation with the current motor") that no amount of gain-tuning can get around.

That last point turned out to be the most useful outcome of the project: a fitted model plus a bit of algebra told us where the *ceiling* on performance was, before we spent time chasing it in hardware code. We had to build a different system to get the pendulum up from the bottom.

## Core Ideas of the Implementation

The project is organized around three stages:

- Derive the equation of motion from first principles (Lagrangian mechanics)
- Fit its free parameters to real sensor data by numerically integrating the ODE, 
- Use the fitted model to simulate the *full* closed loop `pendulum + car + stepper-motor + sensor + control system` and test strategies and tune parameters.
  
### The data 

The recorded data on which the simulation is then built upon is the pendulum released from 60° and left to swing/decay under gravity and friction, no control input, to isolate the *plant* dynamics before any controller is involved.

<image src="/inverted-pendulum/data.png" controls width="100%">
</image>

The Arduino logs semicolon-delimited lines like:

```
T0000086006;A-00002;V-08141;d02351
T0000095150;A-00004;V-16332;d03692
T0000126673;A-00005;V+05176;d31523
```

`T` is a timestamp (encoder ticks), `A` the angle count from the rotary encoder, `V` and `d` auxiliary channels. 

Parsing strips the field prefixes and sign characters:

```python
with open("./datos_filtrados.txt", 'r') as archivo:
    lineas = archivo.readlines()

t_values, a_values = [], []
for linea in lineas:
    data_values = linea.strip().split(';')
    if len(data_values) == 4:
        t_values.append(int(data_values[0][1:]))
        a_values.append(int(data_values[1][1:]))
        # ...

y = np.array(a_values[117:])          # drop a noisy warm-up window
t = np.array(t_values[117:])
t = (t - t[0]) / 100000               # convert ticks to seconds
```

Further data (pendulum's length, printer's length, step size of the motor, time intervals for measurements and actions from the arduino, sensor resolution etc.) is gathered to make the simulation as close to the real system as possible.     

### Modeling

Rather than fitting a generic damped-oscillator formula, I derived the equation of motion for *this specific system* (car + pendulum) from Lagrangian mechanics
$$
\frac {d^2 \theta} {dt^2} = \frac{\kappa}{g}\, a\cos\theta \;-\; \kappa\sin\theta \;-\; \mu\,\frac {d \theta} {dt} \;-\; \gamma\,\mathrm{sign}(\frac {d \theta}{dt})
$$
(see the [Math](#the-math) section for a derivation and explanation of each parameter)

and fit its physical constants, not by evaluating a closed-form solution, but by **numerically integrating the ODE inside the fit**:

```python
def general_dSdt(S, t, kappa, mu, gamma):
    theta, theta_dot = S
    Dtheta = theta_dot
    Dtheta_dot = (-kappa*np.sin(theta*tau/730)
                  - mu*theta_dot*tau/730
                  - gamma*np.sign(theta_dot*tau/730))
    Dtheta_dot = Dtheta_dot*730/tau
    return [Dtheta, Dtheta_dot]

def angle(t, kappa, mu, gamma):
    dSdt = lambda S, t: general_dSdt(S, t, kappa, mu, gamma)
    sol = odeint(dSdt, S_0, data[0])
    return sol.T[0]

params = curve_fit(angle, t, y, (55, 0.22, 0.3), maxfev=10000)[0]
```

Every call to `angle()` during the optimization re-integrates the nonlinear ODE with the trial parameters and compares the trajectory to the real encoder trace — `curve_fit` is doing least-squares over *simulated trajectories*, not over an algebraic formula. This is slower than fitting a closed-form expression, but it's the only honest way to fit a model whose ODE has no clean analytic solution (the `sign(θ')` term makes it piecewise, and the `sin(θ)` term makes it nonlinear).

Since the data is gathered with the free pendulum, we are actually fitting to this formula: 
$$
\frac {d^2 \theta} {dt^2} = -\; \kappa\sin\theta \;-\; \mu\,\frac {d \theta} {dt} \;-\; \gamma\,\mathrm{sign}(\frac {d \theta}{dt})
$$

<image src="/inverted-pendulum/fitted_model.png" controls width="100%">
</image>

Luckily the constant $\kappa$ (which from the Lagrangian derivation is $mlg/I$ where $I$, m and l are the pendulum's moment of inertia, mass and length respectively) appears in the acceleration term $\kappa /g a \cos(\theta)$ and the gravitational fall $\kappa \sin(\theta)$ so we didnt have to fit a separate parameter for which we would need more data with the car included.

### Simulation

Once the optimal κ, μ, γ are known, the same physics function drives a full discrete-time simulation that adds the car's acceleration as a forcing term and layers hardware realism on top: control-loop latency, finite stepper speed, sensor quantization, and step saturation.

```python
def Dtheta_dot_func(theta, theta_dot, a):
    Dtheta_dot = (-Lambda*a*np.cos(theta*tau/730)
                  - kappa*np.sin(theta*tau/730)
                  - mu*theta_dot*tau/730)
    Dtheta_dot = np.where(
        (np.abs(Dtheta_dot) < np.abs(gamma)) & (theta_dot == 0),
        0, Dtheta_dot - gamma*np.sign(theta_dot)
    )
    return Dtheta_dot / (tau/730)

def PID(theta_int, theta, theta_dot):
    error = resolution(shift(theta, 365), res)
    response_vel = kp*error + kd*theta_dot + ki*theta_int
    return -cap(response_vel, step_size/(car_reaction_time*dt))
```

The main loop advances physics every `dt` (0.1 ms, simple Euler integration) but only re-runs the control logic every `dt_system` (0.5 ms), mimicking the Arduino's actual update rate:

```python
for i in range(N):
    physics()
    if not i % dt_system:
        control()
    append()
    if abs(x) > Max_x:
        print('car fell')
        last = i
        break
```

This separation — fast physics tick, slow control tick — is what lets the simulation expose *timing-related* failure modes (e.g. the controller reacting too late) rather than just idealized continuous-time PID behavior.

## The Math

### 1. Deriving the equation of motion (Lagrangian mechanics)

The pendulum is modeled as a continuous mass distribution ρ(λ) along its length, pivoting on a cart moving horizontally with position *x*. Writing the kinetic and potential energy of a mass element and integrating:

$$
K = \frac{1}{2} m v^2 + \frac{1}{2} I \theta'^2 + v\cos\theta\, \theta' \, m l, \qquad U = -mgl\cos\theta
$$

where *m* is the pendulum's mass, *l* its center-of-mass distance from the pivot, *I* its moment of inertia, and *v* the cart's velocity. Applying the Euler–Lagrange equation,

$$
\frac{\partial L}{\partial \theta} = \frac{d}{dt}\left(\frac{\partial L}{\partial \theta'}\right), \qquad L = K - U,
$$

and simplifying (the details are in the notebook) yields:

$$
I\,\frac {d^2\theta}{dt^2} = mla\cos\theta - mlg\sin\theta
$$

where *a = dv/dt* is the cart's acceleration — this is the driving term: pushing the cart accelerates the pivot, and that acceleration couples into an angular torque through cos θ. Adding viscous and dry (Coulomb) friction and dividing through by *I*:

$$
\frac {d^2\theta}{dt^2} = \frac{\kappa}{g}\, a\cos\theta \;-\; \kappa\sin\theta \;-\; \mu\,\frac {d\theta}{dt} \;-\; \gamma\,\mathrm{sign}(\frac {d\theta}{dt})
$$

Three lumped constants — **κ** (gravity/inertia ratio), **μ** (viscous friction), **γ** (dry friction torque) — absorb all the unmeasured physical details (exact mass distribution, bearing friction, etc.) and become the parameters fit to real data. With *a* = 0 this reduces to the free-pendulum equation used for the initial fit.

### 2. Parameter fitting: analytic-solution fits vs. numerical-integration fits

I deliberately tried fitting the data two different ways to find the best model:

- **`Alternative_models.ipynb`** fits *closed-form analytic expressions* directly — e.g. an exponentially-decaying cosine
  $$
  \theta(t) \approx (\theta_0 + \alpha t)\,e^{-\beta t}\cos(\omega t)
  $$
  These are fast to fit (`curve_fit` just evaluates the formula) and fine as a sanity check, but they're phenomenological — the parameters α, β, ω don't map onto physical quantities like mass or friction coefficient, and the model can't be extended to include a forcing term for the cart.
- **`Model_fitting_and_simulation.ipynb`** fits the *ODE itself* by numerically integrating it (`odeint`) inside the objective function passed to `curve_fit`. This is more expensive per iteration, but κ, μ, γ are physically meaningful and directly reusable in the forced (cart-driven) simulation — the whole reason the more expensive route is worth it.

The fitted free-pendulum parameters came out to:

```
kappa ≈ 56.62      (g-scaled restoring-torque constant)
mu    ≈ 0.220      (viscous friction coefficient)
gamma ≈ 0.300       (dry friction torque)
```

### 3. PID control

A standard PID acting on the angular error from the inverted equilibrium (365 encoder counts in this rig's units), producing a *target car velocity* that's translated into a wait time between stepper steps:

$$
u(t) = k_p e(t) + k_i \int_0^t e(\tau)\,d\tau + k_d \dot e(t), \qquad e = \theta - \theta_{\text{eq}}
$$

implemented with an anti-windup cap on the integral term (`Max_int`) and a saturation cap on the commanded velocity to respect the stepper's maximum speed — both necessary when simulating real actuator limits instead of an idealized force input. (As noted in the code comments, the derivative term wasn't fully tuned yet at the time of writing.)

### 4. How far can the controller recover from? (closed-form limit)

Beyond gain tuning, I derived an analytic bound on system performance directly from the fitted model. Linearizing near the inverted equilibrium (θ ≈ π, small deviations dθ) and looking at the angle change produced by a *single* motor step of size Δx:

$$
\Delta\theta \approx -\frac{\kappa}{g}\Delta x
$$

Comparing the angle *gained* per step against the angle *lost* to gravity during the wait time *T* between steps (accumulated over *N* steps) and solving for the point where the two balance gives a quadratic in the number of steps *N* needed to reach equilibrium:

$$
\left(\frac{\kappa^2 T}{2g} - \kappa T\right)N^2 + \frac{\kappa^2 T}{2g}N - \frac{1}{T} = 0
$$

Solving for the positive root and converting steps back to degrees gives the **maximum deviation the PID system can ever recover from**, independent of how good the PID gains are — a hard ceiling set by motor speed, step size, and the fitted physical constants.

This limit doesnt mean that the system is doomed for angles less than this limit, but that a simple PID system working against gravity cant recover after that point,  a different system is needed to get the pendulum to the top, one that takes advantage of the energy given by gravity instead of fighting against it. Two modes would have to be in place, `swing_up` to give the pendulum energy and get it inside the *safe zone* and `PID`, once its inside the safe zone, to keep it there. 

After testing different strategies in the simulation (sinusoidal oscilation at the natural frequency of the pendulum, energy based control, etc) and the best strategy we found was a simple left-right swinging, at max speed, and only changing direction when the pendulum crosses the zero angle at the bottom. This system adds energy to the system, getting the pendulum to higher and higher angles, if left alone, it will overshoot and make the pendulum rotate, so it needs a threshold after which it gives control to the PID system to keep the pendulum up high.

The threshold cant be just the maximum angle found before, since angular speed can help a larger angle get closer, and a very high angular speed, might make a "safe" angle impossible to recover from and overshoot, especially since the derivative gain in the PID was not working well in the real system. An energy threshold works very well in this case, The safe zone of energy can be calculated from the safe angle threshold: 
With angle $\Delta \theta_{threshold}$ or less, and with no angular velocity, the control system can recover, the energy the pendulum has at that point is 
$$E_{min} = \kappa \cos(\Delta \theta_{threshold})$$
The pendulum at the top with no angular velocity has energy:
$$E_{top} = \kappa$$

$E_{min}$ is the lower bound on energy for the pendulum to be able to be controled, its either in the safe angle zone or has enough kinetic energy to get there, the upper bound is energy that will push the pendulum past the safe zone on the other side and overshoot. If we define $\Delta E_{threshold}$ as $E_{top} - E_{min}$ then the upper bound is $E_{top} + \Delta E$.

So the safe zone is 

$$E_{min}<E<E_{max}$$
or 
$$|E_{top} - E| < \Delta E_{threshold}$$

We found in the simulation and in the physical setup that this two mode control system with this threshold was able to succesfully invert the pendulum. 

## Results

Plugging the fitted κ and the rig's actual step size / stepper timing into the max deviation formula:

```
maximum recoverable deviation ≈ 6.0 degrees (≈ 12 encoder lines)
```

Running the full discrete-event simulation confirms this almost exactly: starting the pendulum at encoder count 353 (12 lines from equilibrium at 365) the controller recovers it; starting at 352 (13 lines off) it falls. 

<image src="/inverted-pendulum/max-dev.png" controls width="100%">
</image>

With respect to the gains: 
- The proportional gain is the simplest and essential to the system, we found that a high enough value is able on its own to get the pendulum up, but not enough to mantain it there.  
- For the derivative we found our sensor was not precise/fast enough to use the measurements of the angle for the derivative calculation to be useful in the PID, the small range on which the PID can act gave very noisy values for the derivative, but on larger trends like in the `swing_up` mode (much more consistent behavior), the derivative calculation was more accurate and was useful in calculating the energy threshold. 
- For the integral, we found it essential to keep the pendulum up, the proportional gain was able to succesfully get the pendulum up, but due to the 1 encoder line resolution, it has a dead zone over which it cant respond, but the pendulum isnt in equilibrium so it starts falling again, whenever it passes the encoder, the pendulum pushes it back up, but the physics of the system make it conserve the angular velocity it had while falling, meaning the system is gaining energy everytime the car pushes the pendulum back up, leading to an eventual fall once enough energy accumulates. Another problem that arises when no integral is used is due to the error always staing on one side, the corrections done by the car are always to one side aswell, meaning the car would reach the limit of the printer track and the pendulum falls.
<figure>
    <figcaption>The system without an integral gain.</figcaption>
    <image src="/inverted-pendulum/without-integral.png" controls width="100%">
    </image>
    <figcaption>The system with an integral gain.</figcaption>
    <image src="/inverted-pendulum/with-integral.png" controls width="100%">
    </image>
</figure>

The conclusion drawn from combining the closed-form bound with the simulation: **stepper power and the printer's finite track length (the car running off the end) are the binding constraints on this rig**, not sensor resolution or control-loop latency, which turned out to be comfortably fast relative to the pendulum's dynamics. This information helped us make smarter decisions to get the result we wanted, we created two separate systems (swing_up and PID) and we changed a previous stepper motor's driver which only gave us half the power meaning the maximum deviation was around 2.5 degrees giving a very small safe zone for the PID.

## Takeaways

- Deriving the plant model from mechanics first, rather than curve-fitting an arbitrary function, paid off directly: the physical parameters (κ, μ, γ) transferred cleanly from the *unforced* free-pendulum fit into the *forced* car-driven simulation, which a purely phenomenological fit couldn't have done.
- Simulating the discrete, latency-laden, quantized version of the control loop — not the idealized continuous one — was what surfaced the real limiting factors (motor speed and track length), potentially saving a lot of blind gain-tuning on the physical rig.
- A closed-form limit derived from the fitted model gave a sanity check that no amount of PID tuning could ever exceed, which forced us to create a separate system to swing the pendulum up to the safe zone for the PID to act.
- A common issue of the car reaching the limits of the printer length because it keeps moving to one side after succesfully controlling the pendulum can be mitigated by a well tuned integral gain.

---

*Full code, the raw sensor log, and both notebooks are available in the [GitHub repository](https://github.com/juangriffin121/Data-Model-PID-Sim).*
