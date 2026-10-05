---
layout: ../../layouts/MarkdownLayout.astro
title: 'Blobworld: Evolution simulator'
description: 'Evolution simulation in Rust'
tags: ["Rust", "Computational simulation"]
---
# Blobworld: Evolution simulator 

*A from-scratch artificial life simulation in Rust, where blobs hunt, flee, and evolve.*

<video src="/evolution/evolution(seed12).mp4" controls width="100%">
  Your browser does not support the video tag.
</video>

---

## Introduction

Blobworld is a small artificial life simulator written in Rust. It places two populations of simple creatures: **prey** and **predators**, in a shared *toroidal* 2D world. Each creature ("blob") is controlled by its own tiny neural network, and none of them are trained in the conventional sense. There's no backpropagation, no loss function, no gradient descent. Instead, behavior emerges the old-fashioned way: **reproduction, random mutation, and survival**.

Blobs that avoid being eaten (or that successfully hunt) can live long enough to reproduce, passing a slightly mutated copy of their neural network on to their offspring. Over thousands of simulated steps, the population's collective "brains" drift toward strategies that work — chasing, fleeing, circling — without anyone ever telling them what those strategies should look like.

## Motivation

As a machine learning enthusiast with a degree in Biotechnology and Molecular Biology, my interest in this project is twofold: 

- Creating a toy universe for an example of evolution by natural selection.
- Trying a largely unused training method in machine learning and reinforcement learning, *genetic algorithms*, and testing its capabilities in this simulated world.

Most of the neural networks I'd worked with before were trained with gradient descent: define a loss, backpropagate and ideally converge to a good minimum. That's the standard for most neural networks. Evolutionary approaches are a much less common but very interesting and elucidating optimization methods. A network's weights change only because *slightly different weights survived and reproduced slightly better*. There's no gradient telling a blob "turn 3 degrees more toward the prey", there's only "did this blob's descendants outcompete the others."

Building this from scratch in Rust was also a deliberate choice. Simulating hundreds of blobs, each computing vision, running a forward pass, and updating position every single step, is exactly the kind of tight, allocation-heavy inner loop where Rust's performance and `rayon`'s data parallelism actually matter, and where I wanted to build the primitives (matrix multiplication, mutation, ray-based vision) myself rather than reach for a framework.

## Core Ideas & Implementation

### Architecture at a glance

The project is organized as a handful of focused modules:

```
src/
├── main.rs           # CLI entry point: load -> evolve -> save
└── mods/
    ├── world.rs       # the simulation loop itself
    ├── blobs.rs       # blob state, movement, vision, reproduction
    ├── brains.rs      # the neural network: forward pass + mutation
    ├── activations.rs # sigmoid / relu / leaky relu / tanh
    ├── constants.rs   # all tunable parameters, loaded from JSON
    └── utils.rs       # matrix math and geometry helpers
```

Everything a run needs, world size, mutation rate, energy costs, population sizes etc. lives in a single `constants.json`, so I can tune the evolutionary pressure without recompiling:

```json
{
  "seed": 12,
  "reproduction_distance": 1.0,
  "food_energy": 1.0,
  "step_size": 0.5,
  "neuron_length": 20.0,
  "world_shape": [341.5, 192.0],
  "input_neurons_num": 30,
  "motion_energy_cost": 0.007,
  "prey_base_energy_gain": 0.03,
  "predator_base_energy_loss": 0.004,
  "mutation_rate": 0.2,
  "ages": 2000,
  "num_predators": 50,
  "num_prey": 100,
  "max_speed": 5.0,
  "max_angle_diff": 0.3,
  "graph_neurons": false,
  "activation": "relu",
  "render_every": 0,
  "dump_frames": true
}
```

### The brain: a minimal feedforward network

Each blob's brain is deliberately simple: a `Vec` of weight matrices, no bias terms, no framework:

```rust
pub struct Brain {
    pub network_shape: Vec<i32>,
    pub weights: Vec<Vec<Vec<f32>>>,
    pub neuron_angles: Vec<f32>,
    pub neuron_length: f32,
    pub activation: Activation,
}

pub fn synapse(&self, stimuli: &Vec<f32>) -> Vec<f32> {
    let mut input = stimuli.clone();
    for layer in 0..self.network_shape.len() - 1 {
        input = matrix_prod(&self.weights[layer], &input);
        if layer != self.network_shape.len() - 2 {
            input = input.iter().map(|&x| self.activation.apply(x)).collect();
        }
    }
    input
}
```

The output layer is left un-activated, since its two values are later squashed by `sigmoid` (speed) and `tanh` (turning) directly in the world's movement step.

### Vision: rays instead of pixels

Blobs don't see images, they see through a small number of **directional rays**, each corresponding to one input neuron. Every neuron has a fixed angular offset from the blob's facing direction; a neuron "fires" (returns `1.0`) if an opposite-type blob's body intersects that ray within `neuron_length`:

```rust
pub fn check_surroundings(&self, world: &World) -> Vec<f32> {
    // ...for each input neuron, cast a ray at neuron_angle
    // and sum up how many enemy blobs it intersects
}
```

This is a nice, cheap approximation of vision: cheap to compute in parallel across hundreds of blobs, and rich enough that the network can, in principle, learn to distinguish "prey straight ahead" from "prey off to the side."

### Evolution: mutation without gradients

There is no training loop in the ML sense. Instead, when a blob accumulates enough energy, it splits into two children, each receiving a **mutated copy** of the parent's weights:

```rust
pub fn make_child(&self, mutation_rate: f32, rng: &mut impl Rng) -> Brain {
    let weights = sum_weights(
        &self.weights,
        &Brain::delta(&self.network_shape, mutation_rate, rng),
    );
    // ...rebuild the Brain with the new weights
}
```

`delta` just fills a same-shaped tensor with independent uniform noise scaled by `mutation_rate`. That's the *entire* learning rule. Whether a mutation helps or hurts is never evaluated directly, it's only revealed indirectly, over time, by whether that lineage keeps reproducing.

### The world loop

Every simulated step (`World::update`) runs the same five-stage pipeline:

```rust
pub fn update(&mut self, age: i32, rng: &mut impl Rng) {
    let stimuli   = self.gather_stimuli();     // 1. see
    let responses = self.gather_responses(stimuli); // 2. think
    self.move_blobs(responses);                // 3. act
    let interactions = self.check_interactions(...); // 4. collide
    self.kills(interactions, &prey_indexes);
    self.base_energy();                        // 5. metabolize
    self.starved();
    self.reproduce_blobs(rng);
}
```

Stimuli-gathering and response-computation are both parallelized with `rayon`, since each blob's vision and forward pass are independent of every other blob's.

### Energy: the currency of survival

Everything a blob does costs or produces energy, and energy is the sole gate on both death and reproduction:

- **Prey** gain a small trickle of energy each step (representing grazing) and lose energy proportional to how fast they move.
- **Predators** constantly lose energy just to stay alive, and only gain energy by catching prey.
- A blob's **radius** is literally `sqrt(energy)` meaning its area is proportional to its energy content, a 2D form of our 3D `Volume`-`Weight`-`Energy storage` relation, well-fed blobs are visibly bigger in the animation.
- Energy above a threshold triggers reproduction; energy below zero means starvation.

## The Math

**Vision, point-to-segment distance.** A ray is defined by a starting point, a direction, and a length. To test whether a blob lies "on" that ray, the vision system projects the target's position onto the ray and clamps the projection to `[0, length]`:

$$
t = \text{clamp}\big((\mathbf{p}_{\text{target}} - \mathbf{p}_{\text{start}}) \cdot \hat{\mathbf{d}},\ 0,\ L\big), \qquad
\mathbf{p}_{\text{closest}} = \mathbf{p}_{\text{start}} + t\,\hat{\mathbf{d}}
$$

A neuron fires if $\lVert \mathbf{p}_{\text{target}} - \mathbf{p}_{\text{closest}} \rVert \le r_{\text{target}}$, where $r = \sqrt{\text{energy}}$.

**Forward pass.** For layer $\ell$ with weight matrix $W^{(\ell)}$ and activation $\sigma$:

$$
\mathbf{x}^{(\ell+1)} = \sigma\big(W^{(\ell)} \mathbf{x}^{(\ell)}\big)
$$

applied for every layer except the last, whose raw output $(o_1, o_2)$ is turned into motion by:

$$
\text{speed} = v_{\max}\cdot \text{sigmoid}(o_1), \qquad
\Delta\theta = \theta_{\max}\cdot \tanh(o_2)
$$

**Mutation.** A child's weights are the parent's weights plus independent uniform noise scaled by the mutation rate $\mu$:

$$
W'_{ij} = W_{ij} + \mu \cdot U(-1, 1)
$$

There is no gradient here, $\mu$ controls the *size* of random exploration, and selection (survival to reproduction) does the rest.

**Position update (toroidal wraparound).** The world wraps at its edges, so a blob's position after a step of size $s$ is taken modulo the world's dimensions $S$:

$$
x' = \big((x + s \cdot \cos\theta) \bmod S\big)
$$

## Results

To animate the results of the simulation, there's currently two systems:

- Rust based plotting of the current state at each step using `plotters` with `ffmpeg` to turn the resulting graphs into videos. This was my original system, but its slower `render_every=0` in constants blocks this and its set by default.
- Python based animation using matplotlib. With `dump_frames = true` the Rust simulation dumps all necesary data for graphing the current state into a binary file which is then read in the python script, this system is faster, but I havent implemented neuron plotting for it yet, which might increase animation time.

<figure>
    <video src="/evolution/evolution.mp4" controls width="100%">
      Your browser does not support the video tag.
    </video>
    <figcaption>Animation from the begining with population count graphs, animated with the Python system.</figcaption>
</figure>

<figure>
    <video src="/evolution/blobs.mp4" controls width="100%">
      Your browser does not support the video tag.
    </video>
    <figcaption>Visual neurons plotted in the simulation, animated with the Rust system.</figcaption>
</figure>

To find interesting seeds which can show signs of learning, i implemented a cheaper alternative to the full animation: at every step, the simulation dumps into a csv `[age, prey_count, pred_count, mean_prey_energy, mean_pred_energy]` which is then used in python to create the following graphs: 


<figure>
    <image src="/evolution/seed12.png" controls width="100%">
    </image>
    <figcaption>Population counts and total energy per species for the simulation shown at the begining.</figcaption>
</figure>


<figure>
    <image src="/evolution/seed2.png" controls width="100%">
    </image>
    <figcaption>Population counts and total energy per species for another interesting run, multiple predator-prey cycles.</figcaption>
</figure>


<figure>
    <image src="/evolution/all_runs.png" controls width="100%">
    </image>
    <figcaption>Accumulated graphs from 20 runs of the simulation from 20 different seeds.</figcaption>
</figure>

<figure>
    <video src="/evolution/Chase.mp4" controls width="100%">
      Your browser does not support the video tag.
    </video>
    <figcaption>Instance of predator chasing a prey captured in the simulation at the begining, as soon as the predator catches the prey it reproduces, since after eating the prey it passes the energy threshold.</figcaption>
</figure>


Some notable things in the results:

- **Population oscillations:**  In most runs a few predator-prey cycles can be seen, a predator booms crash the prey population, which in turn starves the predators, letting prey recover, Lotka-Volterra modeled this behavior as:
    - $$\frac {dx} {dt} = x(\alpha - \beta y)$$
    - $$\frac {dy} {dt} = -y(\gamma - \delta x)$$
    - This equations could be used to fit to the simulation results and help find better simulation parameters through researching the equation parameters.
- **Extinction:** Most runs end in a predator extinction, usually, one of the cycles previously described, the prey population crash leads to the predators starving but the prey doesnt recover fast enough for the last predators to find them in the now desolate ecosystem. Since the prey dont require encountering its energy, it just gets it constantly, its population is more resilient, a decrease in predator base energy could improve the number of cycles the predator population survives.
- **Emergent strategies:** Predators show signs of learning in the later stages of the simulation, pursuit of prey can be seen, the issue is however that once prey leaves a predator's field of vision the predator lost the chase and wont keep pursuing, this is because the response is only to the current state, this behavior can be improved upon with some tweaks to the neural networks for the blobs. Prey dont seem to learn to avoid predators, the constant flow of energy seems to be enough for the prey to survive without learning survival strategies, also visual neuron separation might limit its field of view meaning the most dangerous predators(those chasing them from behind) arent visible, this property could also be changed to allow for mutation and together with a decrease in energy gain could result in better simulations.
- **Serrated/periodic population graphs:** Especially in prey, a periodic behavior, with a period smaller than the population cycles can be seen, this is due to the time it takes a recently birthed prey to get enough energy from its constant gain to be able to reproduce, this behavior is most notable in the begining of the simulation because all blobs start with the same energy, as the simulation progreses, the different lives the blobs live make this behavior less pronounced. This is not seen in energy graphs since a reproduction event doesnt change the total energy.   

Some future improvements:

- **Neural network changes**: 

    - More hidden layers: Currently the forward pass involves only one matrix multiplication, adding more layers can allow blobs to perform deeper processing of their inputs.
    - Adding internal information to the inputs, a blob that knows its energy reserve can learn to hibernate or to chase faster to avoid starvation depending on base energy spending.
    - Adding previous state to its inputs, it can then learn to "remember" if it was following a prey and lost it, or even learn their relative velocity directions to better intercept them. 
    - Implementing mutation for neuron separation.
- **Parameter space exploration**:  The simulations tend to end around 1000 steps, which in the longer end of the spectrum of length allows for some noticeable learning but longer runs would sharpen these behaviors(especially if we increase the complexity of the blobs' brains), finding the right parameter configuration is the biggest lever we can push on this front.

---

*Code and full implementation: [github.com/juangriffin121/evolution](https://github.com/juangriffin121/evolution)*
