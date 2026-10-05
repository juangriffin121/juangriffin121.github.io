---
layout: ../../layouts/MarkdownLayout.astro
title: 'President: Reinforcement Learning from Scratch'
description: 'Game dynamics analysis with reinforcement-learning agents as players'
tags: ["Python",  "Reinforcement learning"]
---

*Code: [github.com/juangriffin121/President](https://github.com/juangriffin121/President)*

## Introduction

[President](https://en.wikipedia.org/wiki/President_(card_game)) is a shedding-style card game in which players try to get rid of their cards before everyone else. On each turn, a player must either play a set of cards of the same count as the previous play(if there is any) that beats it, or pass, and if everyone passes on a play, the player who played it starts a new round. The first two players to empty their hands become President and Vice President; the last two players become Vice Scum and Scum. Those roles matter in the next game because the high-ranking players can exchange their worst cards for the low-ranking players' best.

This project implements a full game engine for the game and then trains a population of reinforcement-learning agents to play it, with the neural network written entirely by hand in NumPy, no PyTorch, no TensorFlow.

The repo contains a rules engine, a set of hand-coded heuristic bots, a small terminal UI for playing against them and testing their strategies, and an `rl` package with four different card-choosing policies (a linear policy, an MLP, a bilinear "state scorer," and an actor-critic agent) plus a small separate learned policy for the president/scum card exchange, all trained with REINFORCE-style policy gradients.

## Motivation

This project started from noticing something odd about President while playing it with my family. It's a very unfair, luck-dependent game — presidents and scum tend to stay presidents and scum, for the obvious reason that the card exchange keeps stacking the deck — but there's also a subtler, purely positional effect underneath that. Whoever opens a round tends to open with a low card, and the president is the player most likely to open, so the seat right after the president gets a quiet advantage: more chances to win cheap rounds and climb, sometimes even overtaking the president outright. The seat right before the president gets the opposite treatment — their biggest cards are exactly the ones most likely to get beaten by the president's last play, so their role tends to degrade over time. Play that out over many hands and you get a kind of loop around the table: the "second in command" rises until they dethrone the president, at which point their old neighbor is now sitting right behind the new president, primed for the same slow climb, and so on.

I wanted an engine that would let me analyze that trajectory, but for that I needed to simulate players. Around the same time I was reading *Sutton & Barto's Reinforcement Learning: An Introduction*[^1] and wanted an excuse to implement the ideas myself rather than just read them, and I already had the neural-network machinery — `Linear`/`Activation`/backward passes — sitting in my [digit recognizer from scratch](/projects/digit-recognizer) project, which I'd built the same way, by hand, to further my understanding of backpropagation and neural networks. That project taught me the mechanics of gradient descent on a fixed, labeled dataset. Reinforcement-learning with neural networks was a natural idea for the players of the game engine.

President is an interesting problem for a reinforcement learning project. The action space is combinatorial (any same-rank subset of your hand, plus jokers) and state dependent (you must beat the last hand and with the same number of cards), the state is partially observed (you don't know your opponents' hands), luck influences the results havily, and the only real signal you get is *how you placed* at the end of the game. Working on this project taught me the different kinds of policy-gradient methods are built for — and it meant I could reuse almost all of the `Linear`/`Activation`/backward-pass machinery from the digit recognizer, just bolted onto a different kind of loss.

I test the seat-position hypothesis directly, using logged simulation data from a trained agent population — see Results below.

## Project Structure

```
president/
  card.py              # Card and Joker types
  deck.py               # Deck construction + shuffle/deal data
  rules.py              # Move validation rules
  ranking.py             # Card/order helpers
  utils.py               # Legal-move generation (possible_sets)
  state.py               # GlobalState / PlayerState dataclasses
  table.py               # Round/game loop and role transitions
  player.py              # Player state + strategy hooks
  strategy.py            # Human/bot/agent strategies
  play.py                # Interactive game entry point
  ui/                    # Terminal read/write helpers
  nn/
    layers.py            # Linear / activation layers (shared with my digit recognizer)
    network.py           # Generic feed-forward network container
  rl/
    agent.py             # Agent: wraps a card-chooser + a worst-chooser, trajectories, save/load
    chooser.py           # CardChooser interface + WorstChooser (learned exchange policy)
    card_choosers.py     # Linear / MLP / StateScorer / ActorCritic card-choosing policies
    features.py          # State/action feature extraction
    hand_strength.py     # Hand-strength predictor
    train.py             # Training/evaluation loops
    fight.py             # Agent-vs-agent matches
  experiments/
    agent_family.py       # Population-based training (checkpoints, replacement, exchange ablations)
    positional_dynamics/   # Logs + analysis testing the seat-position hypothesis
tests/                    # Pytest suite
```

## Core Implementation Ideas

### The game engine

The rules of President are simple to state but a little fiddly to implement
correctly: a play must match the size of the last play, be a single rank (jokers can
pad any rank), and outrank it. Aces are the top rank, not the bottom:

```python
HIGHEST_RANK = 13

def order_num(num: int) -> int:
    return HIGHEST_RANK if num == 1 else num
```

Legal-move generation is the other fiddly bit — a hand of $n$ cards can be split into
same-rank groups in many ways, and jokers can extend *any* of them. I generate the
full legal move set lazily with a generator rather than materializing every subset:

```python
def possible_sets(hand: list[Card | Joker]) -> Iterator[list[Card | Joker]]:
    cards = [c for c in hand if not isinstance(c, Joker)]
    jokers = [c for c in hand if isinstance(c, Joker)]
    nums = sorted(set(c.num for c in cards), key=order_num)

    for n in range(1, len(hand) + 1):
        for num in nums:
            matching = [c for c in cards if c.num == num]
            for k in range(min(len(jokers), n) + 1):
                needed = n - k
                if needed <= len(matching):
                    for combo in combinations(matching, needed):
                        yield list(combo) + jokers[:k]
    for n in range(1, len(jokers) + 1):
        yield list(jokers[:n])
```

This is the function every strategy — heuristic or learned — filters down to a legal
action set with `valid_choice`.

The game loop (`Table.game`) deals the cards, runs the exchange, and then plays rounds until only one player still holds cards; that player is the **scum**. A round ends when everyone else passes on a play (the player who made it opens the next round) or when a player empties their hand. Finishing order sets next game's roles, including the rule that makes President genuinely unfair: the scum must give the president their two best cards (in my implementation jokers can't be taken), and the president hands back any two cards of their choosing, usually their weakest. The vice-president and vice-scum swap one card the same way. That asymmetry turns out to matter a lot (more on this in Results).

### Strategy pattern, four layers deep

`Table` is the god object in this project and a `Table` holds a list of `Player` objects
`Player` is an abstract implementation and it can have multiple ways of choosing 
its actions, for this project, a strategy pattern seemed like the best option.
`Player` never makes a decision itself — it just holds a `Strategy` and forwards to
it (`choose_cards`, `choose_worst`, `observe_hand`, `inform_of_results`). This is a
plain Strategy pattern: `Player` is the context, `Strategy` is the abstract
interface, and any of the concrete strategies below can be swapped in without
`Player` or `Table` caring which one it got:

| Strategy | Behavior |
|---|---|
| `Pass` | always passes, never plays a card |
| `Smallest` | heuristic — plays the smallest legal beat of the last play (or opens smallest) |
| `Random` | samples uniformly among all legal moves, including passing |
| `UserStrategy` | prompts the terminal for card indices from a human player |
| `AgentStrategy` | delegates the decision to a wrapped `Agent` (RL policy) |

`AgentStrategy` is a `Strategy` like any other as far as `Player` is concerned —
`choose_cards`/`choose_worst` just forward straight to `self.agent`. What it adds is
the RL-specific bookkeeping *around* that decision: pairing it with a
`HandStrengthPredictor`, and, critically, calling `agent.update(reward)` (or
`agent.record_game(reward)` when batching) whenever `inform_of_results` fires at the
end of a game — that's the only point where the environment ever hands out a reward.

`Agent` doesn't do any scoring itself either. It delegates "which cards to play" to
a `CardChooser` — one of the four from the table below — and "which cards to give
away in an exchange" to a separate `WorstChooser` (this mechanic is only used 
by the president and the vice-president, since they can choose which cards they 
*consider* worst , scum and vice-scum are forced to give their highest ranked except 
`Jokers` since they can *pass* as any card), while keeping for itself only what
every RL policy needs regardless of chooser: collecting the trajectory of
`(features, choice_idx, probs)` tuples across a game, freezing/unfreezing for
evaluation, and saving/loading checkpoints.

So the same has-a pattern repeats at three nested scales — `Player` has-a
`Strategy`, `AgentStrategy` has-a `Agent`, `Agent` has-a `CardChooser` and a
`WorstChooser` — and at each level the outer object only ever depends on the
abstract interface below it, never on which concrete implementation is plugged in.
That's what lets `python -m president.play` seat a human, a heuristic bot, and a
specific trained agent architecture at the same table without `Table` or `Player`
ever needing to change.

### Feature engineering

Since I'm not using convolutions or embeddings, the agents see a hand-designed
feature vector rather than raw cards. There are three groups of features:

- **Hand features** — how many cards you're holding, how many pairs/triples/quads
  you have, joker count.
- **State features** — bin composition, how many cards opponents are holding, how
  many seats away the last player is from you, how many opponents are down to 1–2
  cards (this is the "danger" signal — someone is about to win).
- **Action features**, computed per legal move — does this move break up a
  larger set you could've saved, does it dump a joker unnecessarily, how much of
  your hand would remain after playing it, and how many unseen higher responses
  are still out there (a rough estimate of how "safe" the move is).

  All features are normalized to the interval [0, 1] by dividing the quantity by the max value they might have
```python
def get_action_features(action, hand, hand_count, last_play, last_play_rank,
                         seen_rank_counts, seen_jokers) -> list[float]:
    ...
    has_non_joker = any(isinstance(c, Card) for c in action)
    if has_non_joker:
        rank = get_num(action)
        rank_in_hand = sum(1 for c in hand if isinstance(c, Card) and c.num == rank)
        breaks_larger_set = int(action_count < rank_in_hand)
```

### Neural network machinery (shared with my digit recognizer)

`nn/layers.py` and `nn/network.py` are almost a direct port from the digit
recognizer: a `Layer` base class with `forward`/`backward`/`clone`, a `Linear` layer
that does manual matrix-calculus backprop, and `Leaky_Relu`/`Tanh` activations. Two
things were added on top of the original digit-recognizer version: `clone(perturb_std)`
— Gaussian-perturbed weight copies, used for evolutionary-style agent replacement in
population training — and an `accumulate` flag on `backward()` that lets several
games' gradients be summed before a single weight update, instead of always applying
the update immediately:

```python
def backward(self, grad_output: ndarray, dt, cache: ndarray, accumulate: bool = False):
    input_ = cache
    grad_input = self.pesos.T @ grad_output
    if not self.frozen:
        grad_pesos = grad_output @ input_.T
        grad_sesgos = grad_output.sum(axis=1, keepdims=True)
        if not accumulate:
            self.pesos -= grad_pesos * dt
            self.sesgos -= grad_sesgos * dt
    if accumulate:
        return grad_input, (grad_pesos, grad_sesgos)
    return grad_input
```

That `accumulate` path is what makes the batched training described below possible.

### Policy representation

Every card-chooser reduces to the same shape of problem: given a state and a list of
legal actions, output a probability distribution over those actions $P(a|s) = \pi_{\theta}(a, s)$, sample one, and
later receive a scalar reward for the whole game. The output layer is always a
temperature-scaled softmax over per-action scores:

```python
def softmax(x: ndarray, temp: float) -> ndarray:
    x = x / max(temp, 1e-6)
    max_x = np.max(x) if x.size else 0
    exp_x = np.exp(x - max_x)
    total = exp_x.sum()
    return exp_x / total if total > 0 else np.zeros_like(exp_x)
```

Four card-choosers differ only in how they compute those scores:

| Card chooser | Scoring function |
|---|---|
| `LinearChooser` | single weight vector dotted with concatenated state+action features, simplest setup and the first one I built |
| `MLPChooser` | hand-rolled multi-layer perceptron over the same concatenated features |
| `StateScorerChooser` | bilinear form — a matrix maps state features to a per-action-feature weight vector |
| `ActorCriticChooser` | separate state/action encoders into a shared latent space, scored by dot product; a critic head estimates state value |

Each of these implements the shared `CardChooser` interface mentioned above
(`get_probabilities`, `update`, `save_payload`/`load`, `clone`, ...). `WorstChooser`
is simpler still — just a one-layer network — but it *learns* which cards to hand
over during the president/scum and vice-president/vice-scum exchanges, rather than
always giving away the literal lowest-ranked cards.

## The Math

### Setting up the policy

At each decision point the agent has a state $s$ (hand, table, opponents' hand
sizes...) and a set of legal actions $\mathcal{A}(s)$. A scoring function
$f_\theta(s,a)$ (linear, MLP, or bilinear, depending on the chooser) assigns each
legal action a score, and the policy is a temperature-scaled softmax over those
scores:

$$\pi_\theta(a \mid s) = \frac{\exp\big(f_\theta(s,a) / \tau\big)}{\sum_{a' \in \mathcal{A}(s)} \exp\big(f_\theta(s,a') / \tau\big)}$$

$\tau$ is annealed downward over training (alongside the learning rate `dt`), so early
games explore more and later games commit more strongly to whatever the policy has
learned.

### The policy gradient theorem

The objective is to maximize expected total reward over a trajectory
$T = (s_0, a_0, s_1, a_1, \dots)$ sampled from the policy:

$$J(\theta) = \mathbb{E}_{T \sim \pi_\theta}\left[\sum_{t} r_t\right]$$

The policy gradient theorem says this gradient can be estimated without ever
differentiating through the environment's (non-differentiable, unknown) dynamics:

$$\nabla_\theta J(\theta) = \mathbb{E}_{T \sim \pi_\theta}\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t\right]$$

where $G_t$ is the return from time $t$ onward. Intuitively: nudge the log-probability
of every action you took in the direction that would have made it *more* likely,
weighted by how good the outcome turned out to be.

Subtracting any state-dependent baseline $b(s_t)$ doesn't bias this estimator (its
expectation under the policy is zero), but it can dramatically reduce variance:

$$\nabla_\theta J(\theta) = \mathbb{E}_T\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\big(G_t - b(s_t)\big)\right]$$

### REINFORCE, as implemented here

President only pays out a reward at the very end of a hand — placement performance
$\{-2, -1, 0, +1, +2\}$ for scum through president — so every action taken during
that hand shares the same return $G_t = R$ (no discounting, no per-step shaping). The
`Agent` accumulates a full trajectory of `(features, choice_idx, probs)` across a
whole game and only hands it to the chooser once the terminal reward arrives:

```python
def update(self, trajectory: list[tuple[Features, int, ndarray]], reward: int) -> float:
    if self.frozen:
        return 0.0
    advantage = reward - self.baseline
    self.baseline += self.baseline_lr * (reward - self.baseline)
    for features, choice_idx, probs in trajectory:
        self._update_weights(features, choice_idx, probs, advantage)
    return float(advantage)
```

`self.baseline` is an exponential moving average of past rewards — the simplest
possible variance-reducing baseline, shared across the whole game rather than
computed per state.

The per-step gradient itself comes straight out of the softmax log-derivative. For a
softmax with temperature $\tau$ over scores $z$, and a sampled action $a$:

$$\frac{\partial}{\partial z_i} \log \pi_\theta(a \mid s) = \frac{1}{\tau} (\delta_{ia} - \pi_\theta(i \mid s))$$

Scaled by the advantage, this is exactly `softmax_grad`:

```python
def softmax_grad(y: ndarray, temp: float, choice_idx: int, reward: float) -> ndarray:
    one_hot = np.zeros(y.shape[0])
    one_hot[choice_idx] = 1
    return reward * (one_hot - y) / max(temp, 1e-6)
```

That gradient then gets pushed back through whatever scoring function produced it
(a dot product for `LinearChooser`, a full backward pass for `MLPChooser`), and
weights are updated by **ascending** it — note the `+=`:

```python
self.weights += self.dt * grad @ flat_features
```

**Batched updates.** A single game is a handful of decisions with a reward that only
takes five values, so a per-game update is a very high-variance gradient estimate.
The population-training experiments train with a `batch_size` instead:
`Agent.record_game` stashes several games' trajectories, `Agent.apply_batch()`
averages their gradients through each chooser's `update_batch`, and `WorstChooser`
mirrors this while keeping a *separate* running baseline per exchange size (2 cards
for president/scum, 1 card for vice-president/vice-scum, since the two exchanges have
different reward scales). This doesn't change the math above — it's still REINFORCE
with a baseline — it just averages several independent samples of the gradient
before applying it, trading the speed of per-game updates for a less noisy step.

### Connection to ordinary gradient descent


Understanding this connection was one of my favorite results out of this project. In the digit recognizer example, with softmax + cross-entropy classification with true label $y$, the gradient of the cross-entropy loss with respect to the logits is:

$$\frac{\partial L}{\partial z_i} = \pi(i) - \delta_{iy}$$

Comparing that to the policy-gradient expression above:

$$\frac{\partial \log \pi_\theta(a \mid s)}{\partial z_i} = \delta_{ia} - \pi_\theta(i \mid s)$$

They're the same expression, up to sign and an extra scalar. **REINFORCE is
supervised softmax classification where the sampled action plays the role of the
label, and the per-example loss weight is the return instead of a constant 1.** A
game that ends well retroactively tells the network "the actions you sampled were
the right label, learn them hard"; a game that ends badly tells it the opposite. This
is exactly why the digit recognizer's `Linear`/`Activation`/`backward()` code could
be reused almost verbatim — the forward/backward math doesn't care whether the
"label" came from a dataset or from the agent's own dice roll, only that a gradient
signal exists at the output layer. 

I struggled with the idea of picking one trajectory and assign to every decision the result 
and use that to backpropagate but the reason this works is actually the same reason I didnt
have to use the entire dataset for every backward step in my digit-recognizer to generate the
true gradient, I can use one sample (or a batch) and do a step then go to the next (stochastic
gradient descent) and in expectation the result is the same but it allows the system to go 
through a large dataset (or a very large state-action space) and learn step by step. 


### Actor-Critic: replacing the baseline with a learned value function

A single scalar EMA baseline is crude — it's the same number regardless of whether
you're holding a great hand or a terrible one, this is especially relevant given 
how much this game depends on hand strength. Moreover assigning the same result to 
all decisions is very noisy, if an agent made one mistake at the middle of a game
which costed them the game but otherwise played well, all decisions are considered
and "punished" equally, the initial good decisions are still asociated with the 
result and the decisions done after the mistake were not going to change the result 
because the game was already lost.  

`ActorCritic` instead learns a state-value function $V_\phi(s)$ and uses it as a 
per-state baseline, giving the advantage estimate:


$$A_t = R - V_\phi(s_t)$$

The critic's last layer is a `Tanh`, and its output is scaled by 2, so predictions are
bounded to $[-2, 2]$ — exactly the range the reward can take. It's trained by ordinary
gradient descent on mean-squared error against the realized return:

$$\mathcal{L}_{\text{critic}}(\phi) = \big(V_\phi(s_t) - R\big)^2 \qquad \Rightarrow \qquad \frac{\partial \mathcal{L}_{\text{critic}}}{\partial V_\phi} = 2\big(V_\phi(s_t) - R\big)$$

```python
v, critic_cache = self.critic.forward(ls)
v = 2 * v  # critic ends in Tanh(); this bounds predictions to [-2, 2] like the reward
...
dL_dv = 4 * (value_estimation - reward)  # one 2 from d(2*tanh)/d(tanh), one from d(x^2)/dx
dL_dls = self.critic.backward(dL_dv, dt, critic_cache)
```

while the actor gradient is exactly the same softmax-log-derivative machinery as
REINFORCE, just with $A_t$ standing in for $G_t - b$. 
State and action features each pass through their own small encoder into a shared latent space, and an action's score is the dot product of its latent vector with the state's (a two-tower model). The critic is a small head on top of the state encoder, so actor and critic share that encoder's weights. One naming note: the advantage here uses the full Monte-Carlo return $R$ rather than a bootstrapped estimate, so strictly this is REINFORCE with a learned baseline. Sutton & Barto reserve "actor-critic" for methods that bootstrap from $V$. I kept the name because the actor and critic are trained jointly.

## Results

### Training

I trained agents of all four `CardChooser` types: `LinearChooser`, `StateScorerChooser` (SSA), `ActorCriticChooser` (AC) and `MLPChooser` (MLP64, for its 64-neuron hidden layer). Every training table has four players: the agent against four `Smallest` bots. Rewards run from −2 to +2 and sum to zero at a table, so a player exactly as good as its opponents averages 0.

To reduce noise I batch gradients over several games and train several agents of each type in parallel. At each checkpoint, every agent is frozen, its temperature is dropped to 0.01, and it plays 100 test games on the same fixed set of deals, so all agents face identical luck. The best agent is saved to an `.npz`, and the worst [N] agent(s) are replaced by a noisy clone of it.

Without the card exchange, all agent types plateau at an average reward of about +0.7, a clear edge over `Smallest`.

<figure>
    <image src="/president-rl/spaghetti_curves.png" width="100%">
    </image>
    <figcaption>Training curves and checkpoint tests for the four agent families without exchange</figcaption>
</figure>

With the exchange, they reach about +1.6. Agents first train without exchange, and the exchange is switched on afterwards, which makes it easier for them to learn the basics first.

<figure>
    <image src="/president-rl/spaghetti_curves_all_not_frozen.png"  width="100%">
    </image>
    <figcaption>Training curves and checkpoint tests for the four agent families with exchange</figcaption>
</figure>

The number +1.6 at which the agents plateau needs some care: the exchange is a positive feedback loop (presidents get the scum's best cards and tend to stay president), so part of the jump comes from the mechanic itself rather than learning how to play to that new mechanic. To separate the two, I also ran a baseline where agents are frozen once pretraining ends: 

<figure>
    <image src="/president-rl/spaghetti_curves_all_frozen.png"  width="100%">
    </image>
    <figcaption>Training curves and checkpoint tests for the four agent families with exchange, freezing after turning exchange on</figcaption>
</figure>

The improvement from +0.7 to +1.6 in performance when working with exchange seems to come mainly from learning optimal play under the exchange mechanic rather than just that unfair mechanic rewarding already better strategies.  

In all experiments, all four families plateau at roughly the same level; they differ mainly in training speed and in how often they get stuck in a local optimum.

I expected `LinearChooser` to significantly underperform since from its design its not able to use the `state`+`hand` features its given since they are just concatenated to each action and the dot product can be decomposed into a constant that depends on only the `state`+`hand` which is the same for all action and the action dependent value, and the softmax function used to turn the logits into probabilities is not affected by constants added to all its logits.


We can decompose the learned weight vector into a part that dots with the `state`+`hand` part and one that dots with the `action` part. 

$$ W = [W_{sh}, W_a] $$

For every action, we get the same `state`+`hand` features and concatenate it with that particular action's features. 

$$ F(a) = [F_{sh}, F_a(a)] $$

The logits for each action are then computed as the dot product between those two.

$$ L(a) = W \cdot F(a) $$

We then get the probabilities using the softmax over the logits for all actions.

$$ P(a) = Softmax(\{L(a') ; a' \in A\}) (a) $$

But if we expand the dot product from before:

$$ L(a) = W \cdot F(a) = W_{sh} \cdot F_{sh} + W_a \cdot F_a(a)$$

$$ L(a) = W \cdot F(a) = W_a \cdot F_a(a) + C$$

And due to the softmax's independence from added constants.  


$$ P(a) = Softmax(\{L(a') ; a' \in A\}) (a) = Softmax(\{ W_a \cdot F_a(a') ; a' \in A\}) (a) $$

Its for that reason that `LinearChooser` is unable to use its state features to learn and why I assumed it'd underperform, however, the richness of the features used for the agents mean that the action features seem to be enough to learn the best strategy against `Smallest`, many action features already carry state information (the hand left after the move, how many higher responses remain unseen, how far the move beats the last play). Perhaps with training against other agents the other architectures could outperform it or just turning off certain features would show the limitations of `LinearChooser`

### Seat offset hypothesis analysis

To test the hypothesis I logged 30,000 games with the exchange mechanic, I did multiple runs of this experiment, with different player types(agents with `LinearChooser` and `StateScorerChooser` and players with `Smallest` strategy) and different table sizes.  

The following is an animation of the first 1000 games of a run with `Linear` agents where I show the roles in each game. 

<figure>
    <video src="/president-rl/run5_Linear.animation.mp4" controls width="50%">
      Your browser does not support the video tag.
    </video>
    <figcaption> 5 trained frozen linear agents playing against eachother, with the current president being marked as green, and scum as red 1000 game run</figcaption>
</figure>

The following results come form a 5-player table of frozen `StateScorer` agents (clones of one trained agent, acting at temperature 0.01), five players divides the 50-card deck evenly, so every deal is fair. The results were mostly replicated with the other runs, even the ones without train agents, and the results also remain with tables of number of players other than 5 even when those games arent perfectly fair.

Three main factors for a player affect its performance on a game, the `seat_offset` from the current president, its `current_role_value` and the strength of the hand dealt to them. This last two values are correlated however since presidents and vice presidents tend to have better hands than scum and vice scum due to the exchange mechanic. 

When we plot the mean performance of the players against their `seat_offset` and `current_role_value` we can see that both correlate positively with a higher performance.

Its worth noting the absence of a mean performance value for `seat_offset` $=0$ and `current_role_value` $\neq 2$ since a `seat_offset` of 0 means the player is the president and a `current_role_value` different to 2 means its not the president.

<figure>
    <image src="/president-rl/Performance-Matrix.png" width="100%">
    </image>
    <figcaption>Mean performance plotted as a function of <code>seat_offset</code> and <code>current_role_value</code></figcaption>
</figure>

While `current_role_value` and hand strength are correlated, its not always true that presidents have the best hand, and a hand strength metric can add signal to models that predict performance.

We trained a small network to predict the performance of a player in a game based on its starting hand (after exchange if there was). 


| Performance | Finish role| Mean prediction | Standard deviation |
| ---------------| ----- | --------------- | --------------- |
|$-2$ |   `Scum`| $-0.71$|$	0.82 $|
|$-1$ |   `Vice-scum`| $-0.42$ |$	0.88 $|
|$0$ |	`Commoner`|$ -0.12 $|$	0.91 $|
|$1$ |	`Vice-president`|$ 0.30 $|$	0.91 $|
|$2$ |	`President`|$ 0.89 $|	$0.83$ |

As it can be seen the predictions tend to correlate with performance, but the signal is still noisy.

Linear models predicting performance give R² = 0.29 for hand strength alone, 0.11 for `current_role_value` and 0.09 for `seat_offset`. Hand strength and role overlap heavily (adding role to hand strength only moves R² from 0.29 to 0.32), while `seat_offset` adds information on top of either (hand + seat: 0.33; all three: 0.335). On its own, `seat_offset` is nearly as informative as role. No model is close to perfect, since luck and unobservable factors matter a lot (distance to scum, distance to vice_president, distance to some player with very high strength but not high rank(unobservable for player)).

For context: the president stays president 43% of the time, so the other four seats share the remaining 57%, about 14.4% each. Against that baseline, the vice-president becomes president 19.95% of the time and the player at `seat_offset = 1` 18.55% (almost as much as the vice president!), while offsets 2, 3 and 4 get 14.5%, 12.3% and 12.1%.

To see the distribution of winners of games depending on the `seat_offset`, I plot a bar-chart of the results.

<figure>
    <image src="/president-rl/Presidents.png" width="100%">
    </image>
    <figcaption>  Presidents (game winners) bar chart function of <code>seat_offset</code></figcaption>
</figure>

As expected `seat_offset` $=0$ ie the current president, is vastly overrepresented in the game winners, but `seat_offset` $=1$ is also higher than the rest. 

We also plot the distribution of game losers.

<figure>
    <image src="/president-rl/Scums.png"  width="100%">
    </image>
    <figcaption>  Scums (game losers) bar chart function of <code>seat_offset</code></figcaption>
</figure>

As expected `seat_offset` $=0$ ie the current president, is vastly underrepresented in the game losers, but `seat_offset` $=1$ is also lower than the rest and `seat_offset` $=5-1$ ie the left-hand of the president is overrepresented in the losers. 

To see the winners distribution better, we remove the current president from the picture and only take into account the new presidents. 

<figure>
    <image src="/president-rl/New-presidents.png"  width="100%">
    </image>
    <figcaption> New presidents (game winners that were not presidents before) bar chart function of <code>seat_offset</code></figcaption>
</figure>

To check if the results are statistically significant I made a test against the null hypothesis (uniform distribution of winners over seat offsets, ie seat offsets dont matter).

<figure>
    <image src="/president-rl/New-presidents-stats.png"  width="100%">
    </image>
    <figcaption> Statistical analysis of the bars from the previous plot with respect to the null hypothesis (uniform distribution over seat offsets)</figcaption>
</figure>

As it can be seen `seat_offset` $=1$ is significantly higher than expected under the null hypothesis and `seat_offset` $=4$ and `seat_offset` $=3$ are significantly lower. 

`seat_offset` $=1$ might have a two-fold advantage (my interpretation; I haven't isolated the mechanisms). First, they act right after the president, who tends to open low, so they get more chances to win cheap rounds and shed weak cards early. Second, due to this advantage, they climb in role, often to vice-president, and from there they can overtake the president easier than the rest. The role breakdown fits: 51% of new presidents at offset 1 were vice-presidents beforehand, versus 25–29% at the other offsets.

I segment the bars by role in the next plot to show this.

<figure>
    <image src="/president-rl/New-presidents-roles.png"  width="100%">
    </image>
    <figcaption> Same new-presidents plot but now segmented by <code>current_role</code> (role before winning)</figcaption>
</figure>

The set of winners that had `seat_offset` $=1$ appears to have a larger percentage of vice-presidents than other offsets. Since a president tends to stay in power its reasonable that `seat_offset` $=1$ which has an advantage as was shown would grow in rank.

The same statistical analysis from before can be done on the Scums plot to show that `seat_offset` $=1$ is significantly less likely to become scum and that `seat_offset` $=3$ and `seat_offset` $=4$ are significantly more likely.

<figure>
    <image src="/president-rl/Scums-stats.png"  width="100%">
    </image>
    <figcaption> Statistical analysis of the bars from the Scums plot with respect to the null hypothesis (uniform distribution over seat offsets)</figcaption>
</figure>

Overall we see a pronounced positional effect that is almost as high a signal as the vice roles, this effect is replicated over runs with different agents/strategies and different player numbers. That makes a looping dynamic plausible, but showing it properly needs a Markov-chain analysis over (role, seat) states, which I haven't done. I'd expect only a light tendency: luck is a huge factor, a strong hand at a disadvantaged seat can break a loop, and at larger tables most players are commoners far from the president who barely feel the effect.

[^1]: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). The MIT Press. http://incompleteideas.net/book/the-book-2nd.html
