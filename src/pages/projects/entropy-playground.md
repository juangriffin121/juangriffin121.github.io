---
layout: ../../layouts/MarkdownLayout.astro
title: 'Entropy playground'
description: 'A Rust particle simulator with configurable chemical reactions and atomic bonds, built to make thermodynamics something you can watch happen'
tags: ["Rust", "Computational simulation", "Python"]
---
# Entropy Playground

*A Rust particle simulator with configurable chemical reactions and atomic bonds, built to make thermodynamics something you can watch happen.*

[Repo →](https://github.com/juangriffin121/entropy_playground)

---

## Introduction
Entropy Playground is a 2D particle physics engine written from scratch in Rust. Atoms fly around a box, bounce off walls and each other, form bonds when they collide correctly, and break apart when those bonds are stretched too far. On top of that simple physical picture sits a small chemistry layer: atoms have "chemical nature" (`H`, `C`, `N`, `O`, ...), bonds have per-pair spring constants and breaking thresholds, and molecules are defined once as simple graphs and then automatically expanded into every reaction (forming and breaking) that molecule could ever take part in.

Everything about the world: the elements that exist, how they're distributed, which bonds are possible, which molecules are legal, how many atoms there are, how long the simulation runs, is described in JSON files and loaded at startup. 

The project can seed a simulation two ways:

- **Random**: atoms scattered with random positions and velocities, sampled from an element distribution.
- **From an SVG path**: atoms are placed along the outline of an arbitrary drawing (a word, a shape, a logo), giving you a highly ordered starting configuration instead of a random gas.

That second mode is where the actual point of the project shows up.

---

## Motivation

The second law of thermodynamics is usually taught as an equation (ΔS ≥ 0) or a slogan ("Nature tends towards disorder"). The connection to the microscopic behavior and the information theory definition of entropy is one of the most misunderstood and underexplained subjects in physics and chemistry.

I wanted to build a small physical system where you could **watch** an ordered state fall apart into a disordered one, I also wanted to explore a neat version of this idea which resulted in an interesting visualization: if you take a low-probability, highly ordered configuration and integrate Newton's equations *backward* in time from it, it *also* becomes disordered. The equations of motion don't know which way "forward" is; entropy increases away from special, low-probability states in **both** temporal directions. This is essentially Boltzmann's resolution to Loschmidt's reversibility paradox, and the reason cosmologists talk about a "Past Hypothesis". The arrow of time isn't built into physics, it's built into having started somewhere special.

The entrypoint of the project, `main.rs`, does exactly this: it loads one ordered initial condition (atoms traced along an SVG path) and runs it twice from the same seed, once with `dt < 0` and once with `dt > 0`, and renders both as frame sequences:

```rust
let mut rng1 = StdRng::seed_from_u64(seed);
let mut recipient = Recipient::from_json("simulation_data/recipient.json", &mut rng1);

for i in 0..recipient.iterations {
    recipient.update(recipient.dt); // dt is negative: integrating backward
    if i % recipient.simulation_rate == 0 {
        plot_recipient(&recipient, &format!("plots/{:05}.png", ...), false);
    }
}

let mut rng2 = StdRng::seed_from_u64(seed);
let mut rec2 = Recipient::from_json("simulation_data/recipient.json", &mut rng2);
rec2.dt = -rec2.dt; // now positive: integrating forward
for i in 1..rec2.iterations {
    rec2.update(rec2.dt);
    ...
}
```

Stitched together, the two frame sequences play as one continuous animation that runs *through* the ordered configuration rather than starting from it, disorder on both sides, order only at a single instant in the middle. That single instant is the closest thing this toy universe has to a "beginning of time."

The following is an example of a simulation performed with this software with a tongue-in-cheek but *theoretically* possible configuration of atoms obtained from an SVG image. 

<video src="/entropy/entropy.mp4" controls width="50%">
  Your browser does not support the video tag.
</video>

This simple example shows not only the "tendency towards disorder", but also, given gravity is included into the simulation, a vertical density gradient can be seen after equilibrium is reached, a thermodynamical result which is the reason why air pressure decreases with height on earth. 

The chemistry layer exists to push the same idea one level further: entropy isn't just about *where* particles are, it's also about *how many independent pieces* the system is broken into. A bond breaking turns one rigid unit with fewer translational/rotational degrees of freedom into two independent ones with more, which is itself an entropy-increasing move. 

---

## Core Ideas of the Implementation

### Module layout

```
src/
├── main.rs
└── bases/
    ├── mod.rs
    ├── atom.rs          # Atom struct + Element/Distribution JSON loading
    ├── bond.rs           # Bond struct, spring force, bond registry helpers
    ├── molecule.rs        # Molecule + MoleculeBlueprint, canonicalization
    ├── reaction.rs        # Forming/breaking reaction blueprints and application
    ├── initialization.rs  # Recursive reaction-graph generation (the core algorithm)
    ├── physics.rs         # Integration, collisions, spatial grid, Force trait
    ├── recipient.rs       # "World" struct, owns everything, built from JSON
    ├── loader.rs           # SVG path parsing for shaped initial conditions
    ├── checks.rs           # Cross-referential validation of the JSON config
    └── plot.rs             # Frame rendering with `plotters`
```

### The `Recipient`: a data-driven world

`Recipient` is the simulation's "world" object, it owns every atom, every bond, the molecule registry, the spatial grid, and the list of forces to apply each tick. It's never built from scratch by hand; it's always deserialized from `simulation_data/recipient.json`, which in turn *references* the other JSON files:

```json
{
  "shape": [1000.0, 1000.0],
  "energy": 1000.0,
  "position_file": "inputs/drawing3.svg",
  "iterations": 1000,
  "simulation_rate": 10,
  "dt": -0.05,
  "use_grid": true,
  "references": {
    "elements": "simulation_data/elements.json",
    "distribution": "simulation_data/distribution.json",
    "molecule_blueprints": "simulation_data/molecules.json",
    "bond_blueprints": "simulation_data/bonds.json"
  }
}
```

Each referenced file is a small, independent piece of the "chemistry" of the world:

```json
// simulation_data/elements.json what elements exist, and their physical properties
{
  "data": {
    "H": { "color": [255, 255, 255], "mass": 1.0079, "radius": 5.0 },
    "O": { "color": [255, 0, 0],     "mass": 15.999, "radius": 20.123 }
  }
}
```

```json
// simulation_data/molecules.json the *only* molecules a human has to define by hand
{
  "blueprints": {
    "Water":  { "atoms": ["H", "O", "H"], "bonds": [[0, 1], [1, 2]], "canonical": false },
    "Oxigen": { "atoms": ["O", "O"],      "bonds": [[0, 1]],          "canonical": false }
  }
}
```

Before any of this gets used, `checks.rs` runs a small validation pass, every element referenced in `distribution.json`, `molecules.json` and `bonds.json` must actually exist in `elements.json`, every bond index must be in range, no atom in a molecule can be bond-less. It's the JSON equivalent of a type-checker for the experiment configuration, and it fails loudly before the simulation ever spends a cycle integrating physics on broken data.

### Atoms, bonds, and forces

An `Atom` is a plain physical object plus a bit of chemistry bookkeeping (which molecule it belongs to, and its position *within* that molecule):

```rust
pub struct Atom {
    pub id: WorldAtomId,
    pub mass: f32,
    pub radius: f32,
    pub position: (f32, f32),
    pub velocity: (f32, f32),
    pub molecule_id: Option<MoleculeId>,
    pub id_in_molecule: Option<MoleculeAtomIndex>,
    pub chemical_nature: String,
    pub bonded_atoms: HashSet<WorldAtomId>,
}
```

Forces are pluggable via a small trait, so gravity and inter-atom bond forces are just two implementations registered on the `Recipient`:

```rust
pub trait Force: Send + Sync {
    fn calculate(&self, atom: &Atom, recipient: &Recipient) -> (f32, f32);
}

force_registry: vec![Box::new(Gravity { g: 1.0 }), Box::new(BondForce {})]
```

Each tick, `Recipient::update` follows a clean pipeline: recompute bond forces (and check for breaks) → sum forces per atom → integrate → resolve collisions:

```rust
pub fn update(&mut self, dt: f32) {
    self.update_bonds();

    let forces: Vec<(f32, f32)> = self.contents.iter()
        .map(|atom| self.total_force(atom))
        .collect();

    for (atom, force) in zip(&mut self.contents, forces) {
        atom.update(dt, force);
    }

    CollisionEngine.process_collisions(self);
}
```

### The collision grid 

Checking every atom against every other atom is O(n²), which gets slow fast once you're simulating hundreds of atoms per frame. `Grid` buckets atoms into cells sized to the largest atom radius, so each atom only needs to check its own cell and the 8 neighbors:

```rust
pub fn get_cell(&self, atom: &Atom) -> (usize, usize) {
    let x = ((atom.position.0 / self.cell_size.0).floor() as isize).max(0) as usize;
    let y = ((atom.position.1 / self.cell_size.1).floor() as isize).max(0) as usize;
    (x.min(self.cells.0 as usize - 1), y.min(self.cells.1 as usize - 1))
}
```

A gridless O(n²) fallback (`gridless_process_collisions`) also exists, mostly useful for correctness-checking the grid path against a simpler baseline.

### Chemistry: generating an entire reaction network from three molecules

This is the part of the project I'm most happy with. Rather than hand-writing every possible reaction ("water can split into H + OH", "OH can split into H + O", ...), the code treats each molecule blueprint as a small graph, atoms are vertices, bonds are edges, and **recursively enumerates every possible fragmentation**.

For each bond in a molecule, `initialization.rs` removes that one bond and does a DFS/connected-components pass over the remaining graph to find the resulting fragment(s):

```rust
pub fn recursive_definitions(
    particle: &ParticleBlueprint,
    reaction_registry: &mut HashMap<MoleculeBlueprint, Vec<(FormingReactionBlueprint, BreakingReactionBlueprint)>>,
) {
    // ...canonicalize the molecule, skip if already processed...
    for (bond_idx, bond) in molecule.bonds.iter().enumerate() {
        let remaining: Vec<_> = molecule.bonds.clone().into_iter()
            .filter(|b| b != bond)
            .collect();

        let (fragments, breaking_reaction, forming_reaction) =
            get_reactions_and_fragments(&molecule, remaining, *bond);

        molecule_reactions.push((forming_reaction, breaking_reaction));

        // recurse into whatever fragments this break produced
        match fragments {
            SingleOrPair::One(new_particle) => recursive_definitions(&new_particle, reaction_registry),
            SingleOrPair::Two((p1, p2)) => {
                recursive_definitions(&p1, reaction_registry);
                recursive_definitions(&p2, reaction_registry);
            }
        }
    }
    reaction_registry.insert(molecule, molecule_reactions);
}
```

The fragment-finding step (`get_fragments`) is a straightforward DFS over the bond graph starting from each end of the removed bond, if both ends still reach every atom, the "break" was actually a ring closure and the molecule stays in one piece with one fewer bond; otherwise you get two disconnected fragments (a smaller molecule, or a lone atom).

Every fragment gets **canonicalized**, atoms are re-sorted by `(chemical nature, number of bonds)` before being used as a lookup key:

```rust
pub fn canonicalize(&self) -> (MoleculeBlueprint, HashMap<BlueprintAtomIndex, BlueprintAtomIndex>) {
    let mut indices: Vec<usize> = (0..self.atoms.len()).collect();
    indices.sort_by_key(|&i| (
        self.atoms[i].clone(),
        self.bonds.iter().filter(|(a, b)| a.0 == i || b.0 == i).count(),
    ));
    // ...rebuild atoms/bonds in the new order...
}
```

This is a lightweight, not-fully-general graph canonicalization (it wouldn't distinguish two structurally different isomers with identical atom/degree multisets), but it's enough for the small organic-chemistry-flavored molecules the project deals with, and it means two chemically identical fragments reached via different bond-break orders collapse into a single registry entry instead of duplicating work.

The result: starting from just `Water` and `Oxigen` in `molecules.json`, the engine automatically derives `H-O` (hydroxyl), free `H`, free `O`, and every forward/backward reaction connecting them, a full reaction graph generated from three lines of JSON.

At runtime, these generated reactions are looked up two ways:

- **Forming reactions** are checked on every hard-sphere collision (`check_forming_reactions` in `reaction.rs`), keyed by what the two colliding particles *are* (free atoms or specific positions within specific molecules).
- **Breaking reactions** are checked whenever a bond's stretch exceeds its `breaking_distance` (inside `update_bonds` in `physics.rs`), keyed by the parent molecule and which bond broke.

Applying a reaction means re-indexing atoms into new `Molecule` structs, updating the `molecule_registry`, and keeping every atom's `molecule_id` / `id_in_molecule` pointers consistent, bookkeeping that's mechanical but has to be exactly right, since a stale reference here corrupts every reaction check downstream.

### Ordered initial conditions from SVG

To get a genuinely ordered starting state (rather than a random gas), `loader.rs` pulls path data straight out of an SVG file with a small regex plus the `svg-path-parser` crate, then `Recipient::from_json` walks every path and drops one atom per point (deduplicated against a grid sized to the largest atom radius, so you don't spawn overlapping atoms):

```rust
let re = regex::Regex::new(r#"<path[^>]*\sd="([^"]+)""#)?;
for caps in re.captures_iter(&svg_data) {
    let parsed = parse_with_resolution(&d.as_str(), 1)
        .collect::<Vec<(bool, Vec<(f64, f64)>)>>();
    all_paths.extend(parsed);
}
```

That's the mechanism behind the "trace a drawing with atoms, then let physics take over" mode described above.

---

## The Math

A few different pieces of math are doing the actual work under the hood:

**Newtonian integration.** Each atom is a point mass integrated with simple explicit Euler steps:

$$v_{t+dt} = v_t + a \, dt \qquad x_{t+dt} = x_t + v_{t+dt}\, dt$$

Because `dt` in `recipient.json` can be negative, the same integrator runs the system backward in time simply by flipping its sign, the "run backward, then forward" trick from the motivation section is literally just `dt → -dt`.

**Elastic hard-sphere collisions.** When two atoms touch, velocities are resolved along the collision normal $\hat n$ using the standard 1D elastic-collision impulse, which conserves both momentum and kinetic energy:

$$J = \frac{2 (\vec v_1 - \vec v_2)\cdot \hat n}{m_1 + m_2}, \qquad \vec v_1' = \vec v_1 - J m_2 \hat n, \quad \vec v_2' = \vec v_2 + J m_1 \hat n$$

**Hookean bonds with a breaking threshold.** A bond behaves like a spring around its equilibrium distance $d_0$:

$$F = k\,(d - d_0)$$

but unlike a real spring, it simply ceases to exist once $d$ exceeds a `breaking_distance`, a crude but effective stand-in for a dissociation energy: past that point, no finite spring force can hold the bond together, so it's treated as broken rather than integrated through.

**Entropy and the arrow of time.** The statistical picture behind $S = k_B \ln \Omega$ is that entropy counts the number of microscopic configurations ($\Omega$) consistent with what you can observe macroscopically. An ordered configuration, atoms tracing a specific shape, corresponds to an astronomically small $\Omega$ compared to "atoms spread roughly evenly through the box," so *any* dynamics that don't specifically aim back at that shape will, with overwhelming probability, drift toward higher-$\Omega$ configurations. Since Newton's equations are time-reversible, this drift happens whether you integrate `dt` forward or backward from the ordered state, which is exactly what the two rendered animations show.

Bond-breaking reactions add a second, chemical flavor of the same idea: fusing two free atoms into a molecule removes translational degrees of freedom (two independently-moving particles become one rigid unit), while breaking a bond restores them, so every breaking reaction is, in this same $\Omega$-counting sense, entropically favorable, and every forming reaction has to be "paid for" by the kinetic energy lost in the collision.

**Graph structure.** Molecules are graphs (atoms = vertices, bonds = edges), and the whole reaction-generation step is a repeated apply-DFS-then-canonicalize over that graph, closer to a small piece of cheminformatics than to classical mechanics, but it's what turns "the physics of two colliding spheres" into "the chemistry of a reaction network."

---

## Results


<video src="/entropy/entropy.mp4" controls width="50%">

Your browser does not support the video tag.

</video>

The simulation outputs the position and velocity of every particle at each timestep and saves it as csv files. I then perform a small Python analysis script which converts these trajectories into probability distributions and calculates their entropy.

For a given quantity $X$, such as the particles' $x$-coordinates, the values at each timestep are grouped into 20 bins. This gives an empirical probability distribution

$$
p_{i}(t)
$$

over the possible values of $X$. The entropy plotted below is then the Shannon entropy of this distribution:

$$
S_X(t) = -\sum_i p_i(t)\ln p_i(t).
$$

Thus, the entropy is high when the particles are spread relatively uniformly across the available bins and lower when many particles become concentrated in a smaller region.

### X position

<image src="/entropy/x_position.png" controls width="100%">

</image>

The three panels show different views of the same quantity.

The **top panel** plots the individual $x$-coordinates of all particles as a function of time. The extremely small and transparent points make the density of particles visible: regions where many points overlap appear darker.

The **middle panel** converts those positions into a histogram at every timestep. The vertical axis represents the spatial bins, while the intensity represents how many particles occupy each bin. This makes the redistribution of particles easier to see than the individual trajectories alone.

The **bottom panel** shows the entropy of the corresponding $x$-position distribution.

One interesting feature is the decrease in entropy in the middle of the simulation. This is the point from which the simulation is started, both backwards in time and forward, which matches our intuition of the video in which the order appears from a disordered start, and eventually disappears into disorder once again. 

As the system approaches equilibrium, the $x$-position distribution becomes approximately uniform. This is what we would expect in the absence of a force that preferentially selects one horizontal position over another: once the transient structure has disappeared, there is no reason for one $x$-coordinate to be substantially more populated than another.

### Y position

<image src="/entropy/y_position.png" controls width="100%">

</image>

The same analysis is performed for the $y$-coordinate, but the result is qualitatively different because gravity acts in this direction.

The **top panel** shows the individual vertical trajectories. Several approximately parabolic trajectories are particularly visible. These arise from the familiar motion of an object being accelerated by gravity: an object initially moving upward slows down, reaches a maximum height, and then falls back down. The multiple prominent parabolas are largely a consequence of the initial configuration being generated from SVG text arranged across several separate lines.

The middle panel shows the evolution of the vertical probability distribution, while the bottom panel shows its entropy.

Unlike the $x$-coordinate, the final $y$-distribution does not necessarily become uniform. Gravity gives different heights different potential energies,

$$
U(y) = mgy,
$$

so the system has a preferred statistical distribution over height.

For a system in thermal equilibrium, the probability of finding a particle at height $y$ is proportional to the Boltzmann factor,

$$
p(y) \propto e^{-mgy/(k_BT)}.
$$

Consequently, the density decreases with height. This is the same basic statistical-mechanical reason that atmospheric pressure decreases as altitude increases: particles at higher gravitational potential energy are statistically less numerous.

For an ideal gas this gives the familiar barometric relation

$$
p(y) = p_0 e^{-mgy/(k_BT)}.
$$

The simulation does not need to explicitly impose this distribution. It emerges naturally from the combination of particle motion, gravity, and the available energy of the system.

The X and Y velocities also show the same dispersion phenomenon, final equilibrium and subsequent increase in entropy shown in the positions but are slightly more confusing to analyze.

An interesting consequence is that the amount of energy supplied to the simulation affects how pronounced this gradient becomes. The system receives an explicit amount of energy during initialization in the form of initial velocities of the particles, but it also starts with **intrinsic potential energy** determined by the initial heights of the particles coming from the SVG. With relatively little available energy, the gravitational potential strongly constrains the particles and the final vertical distribution is noticeably concentrated toward lower heights. With more energy, particles can explore a larger range of heights and the final distribution becomes progressively closer to uniform over the simulated region.

This also explains the contrast between the two coordinates:

* **$x$** has no corresponding gravitational potential, so equilibrium tends toward a roughly uniform spatial distribution.
* **$y$** is coupled to gravitational potential energy, producing a density gradient.
* Increasing the available energy makes that gradient less pronounced because a larger fraction of the particles can populate higher-energy states.

### Interpreting the entropy

The entropy curves should therefore be interpreted together with the corresponding distributions rather than in isolation.

A local decrease in $S_X$ or $S_Y$ occurs when the particles become concentrated into a less uniform distribution. In the position plots this can be seen directly as regions where the trajectories become denser and the histogram develops stronger peaks.

At longer times, the system generally approaches a much more stable distribution. The precise equilibrium distribution depends on the forces acting on the particles and on the total energy available to them, so spatial uniformity is expected in some coordinates but not necessarily in others.

> **Caveat**: The entropy calculated here is the Shannon entropy of coarse-grained position and velocity distributions, the thermodynamic entropy of the full system should be computed from the 4D distribution $\rho(x, y, v_x, v_y)$ rather than taking each variable separate (in fact it should consider all particles separately creating a 4ND distribution, but in this case all atoms are independen and the 4D distribution is representative), however for illustrative purposes, plotting a 4D histogram over time isnt feasible and per-variable histograms represent useful visuals. 

> **A note on the reaction system's current state.** The reaction machinery described above, recursive fragment enumeration, canonicalization, and the forming/breaking bookkeeping, works correctly as far as it goes, and it's the piece of this project I'm happiest with algorithmically. What's still unfinished is integrating it cleanly with the rest of the engine: it doesn't yet play well with the SVG-based initial conditions(atoms bunch up together closely at the start without being in a molecule and backward and forward simulations dont look connected) and getting reaction/bond parameters tuned so molecules form and break at physically sensible rates needs more work. For that reason, the demonstrations above mostly run without active reactions: a single hydrogen species, no bonds, which keeps the physics and the entropy analysis clean while that part of the system catches up. Integrating the chemistry layer with the SVG initialization is the next item on the list.

---

## Tech stack

Rust · [`plotters`](https://crates.io/crates/plotters) for rendering · [`serde`](https://crates.io/crates/serde)/`serde_json` for the data-driven config · [`svg-path-parser`](https://github.com/unicodingunicorn/svg-path-parser) + `regex` for SVG ingestion · `rand`/`rand_distr` for stochastic initialization.
