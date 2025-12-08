---
layout: page
title: "Latent-space Planning and Disentangled Control in MiniGrid"
subtitle: "This project involves first training a reward-free JEPA planning model based on Planning with Latent Dynamics Model (PLDM) (Sobal et al, 2025) using BFS optimal + noisy trajectories from the MiniGrid DoorKey 5×5 environment and analyzing its learned latent dynamics. Building on this baseline, I then introduce a disentangled PLDM variant to examine how separating latent factors influences representation quality and downstream planning performance."
---

## 1. Problem Setting and Vanilla World Model

**Environment**: The environment is a MDP, specifically, MiniGrid DoorKey 5x5 (and transfer to Empty, Empty-Random and Obstacles variants). The agent must pick up a key, unlock a door, and reach the goal. Unlike the original paper's continuous navigation environments, this is a discrete setting.

* True (hidden) state: $s_t \in \mathcal{S}$
* Observation: $o_t \in \mathbb{R}^{H \times W \times 3}$ (RGB grid)
* Action: $(a_t \in \mathcal{A})$ (discrete)
* Transition: $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$

**Data**: ~1200 trajectories (80% optimal and 20% random; 24576 subsequences of length 8 after windowing). Observations are 64×64 → 40x40 (downsized) full-observation RGB arrays, actions are discrete (turn left, turn right, move forward, pickup, toggle, done).

**Goal**: The goal is to learn a **latent world model** that supports planning:

1. An encoder (E) maps observations to latent state $z_t$.
2. A latent dynamics model (f) predicts next latent state from current latent and action.

### 1.1 Vanilla Single-Latent Model

The single-latent model uses a single latent vector
$z_t \in \mathbb{R}^{d}$,
with encoder and predictor:
$z_t = E(o_t), \hat{z}_{t+1} = f(z_t, a_t)$.

Training minimizes:

**Dynamics loss** (self-supervised): $\mathcal{L}_{\text{dyn}} = \text{VICReg}(\hat{z}_{t+1}, z_{t+1}),$ where VICReg combines:

* invariance (MSE similarity),
* variance,
* covariance regularization.

So: $\mathcal{L}_{\text{dyn}} = \lambda_{\text{sim}} , | \hat{z}_{t+1} - z_{t+1} |^2$

* $\lambda_{\text{var}} , \mathcal{L}_{\text{var}}(z)$
* $\lambda_{\text{cov}} , \mathcal{L}_{\text{cov}}(z)$.


Planning: I use a CEM planner in latent space to find action sequences $a_t,\dots,a_{t+H}$ that minimize some distance to a goal latent $z_{\text{goal}}$, using repeated application of (f).

**Issue.** In DoorKey, the state has *heterogeneous structure* (agent, door, key, goal). A single vector (z) must entangle:

* agent position/orientation,
* door locked/open state,
* key picked or not,
* goal location & accessibility,
* static layout.

This makes learning, prediction, and planning harder.

---

## 2. Disentangled Latent World Model

I replace the monolithic latent $z_t$ with a **structured latent representation**:

$z_t = \big(z_t^{\text{dyn}},, z^{\text{stat}},, z_t^{\text{obj1}},, z_t^{\text{obj2}},, z_t^{\text{obj3}}\big)$.

### 2.1 Latent Decomposition

* **Dynamic latent** $(z_t^{\text{dyn}} \in \mathbb{R}^{d_{\text{dyn}}})$

  * Encodes agent-centric dynamic state (position, orientation, etc.).
  * Evolves with actions.

* **Static latent** $(z^{\text{stat}} \in \mathbb{R}^{d_{\text{stat}}})$

  * Encodes time-invariant background (walls, fixed layout).
  * Invariant within an episode.

* **Object latents** $(z_t^{\text{obj}k} \in \mathbb{R}^{d_{\text{obj}}})$ for (k = 1,2,3)

  * Encode changing parts of the environment (key, door, goal, etc.).
  * Discover these factors **unsupervisedly** via reconstruction + dynamics pressure.

The encoder can be written as:

$ (z_t^{\text{dyn}}, z^{\text{stat}}, z_t^{\text{obj1}}, z_t^{\text{obj2}}, z_t^{\text{obj3}}) = E(o_t).$

The key architectural constraints:

1. **Predictor updates only $z^{\text{dyn}}$:**
   $\hat{z}_{t+1}^{\text{dyn}} = f(z_t^{\text{dyn}}, a_t)$, while $z^{\text{stat}}$ and $z_t^{\text{obj}k}$ do not receive gradients through the predictor.

2. **Static latent invariance:**
   $z^{\text{stat}}_t \approx z^{\text{stat}}_{t+1} \quad \forall t \text{ in an episode},$ enforced by an invariance loss.

3. **Decoder reconstructs observation from scene latents:**
   $\hat{o}_t = D\big(z^{\text{stat}}, z_t^{\text{obj1}}, z_t^{\text{obj2}}, z_t^{\text{obj3}}\big)$.
   (I have omitted $z^{\text{dyn}}$ from the decoder to force scene factors into these slots; in the implementation, I use `z_stat + z_obj` for recon.)

### 2.2 Loss Functions

Let (E) and (D) be encoder/decoder. Given a pair $(o_t, a_t, o_{t+1})$:

1. **Dynamics loss (VICReg) on $z^{\text{dyn}}$:**
   
   $\mathcal{L}*{\text{dyn}} = \text{VICReg}\big(f(z_t^{\text{dyn}}, a_t),, z*{t+1}^{\text{dyn}}\big).$

2. **Static invariance loss on (z^{\text{stat}}):**
   
   $\mathcal{L}_{\text{inv}} = \left| z^{\text{stat}}*t - z^{\text{stat}}*{t+1} \right|^2.$

3. **Reconstruction loss on pixel space:**
   
   $\hat{o}_t = D\big(z^{\text{stat}}*t, z_t^{\text{obj1}}, z_t^{\text{obj2}}, z_t^{\text{obj3}}\big), \quad \mathcal{L}*{\text{rec}} = \left| \hat{o}_t - o_t \right|^2.$

Total training objective:

$\mathcal{L} = \mathcal{L}_{\text{dyn}} * \lambda_{\text{inv}} , \mathcal{L}_{\text{inv}} * \lambda_{\text{rec}} , \mathcal{L}_{\text{rec}}$, with typical coefficients (good starting point):

- $(\lambda_{\text{inv}} \approx 0.1)$
- $(\lambda_{\text{rec}} \approx 1.0)$

VICReg itself uses:

$ \mathcal{L}_{\text{dyn}} = \lambda_{\text{sim}} , | \hat{z}^{\text{dyn}}_{t+1} - z^{\text{dyn}}_{t+1} |^2 $

* $\lambda_{\text{var}} \mathcal{L}_{\text{var}}(z^{\text{dyn}})$
* $\lambda_{\text{cov}} \mathcal{L}_{\text{cov}}(z^{\text{dyn}})$.

### 2.3 Planning in Disentangled Latent Space

For planning, I constructed a **planning latent** by concatenating the dynamics and object latents:

$z_t^{\text{plan}} = \big[z_t^{\text{dyn}}, z_t^{\text{obj1}}, z_t^{\text{obj2}}, z_t^{\text{obj3}}\big].$

Given a goal observation (o_{\text{goal}}), we obtain:

$z_{\text{goal}}^{\text{plan}} = \big[z_{\text{goal}}^{\text{dyn}}, z_{\text{goal}}^{\text{obj1}}, z_{\text{goal}}^{\text{obj2}}, z_{\text{goal}}^{\text{obj3}}\big].$

The CEM planner optimizes sequences of actions $(a_t,\dots,a_{t+H-1})$ to minimize cumulative distance:

[
\text{cost}({a_\tau})
= \sum_{\tau=t}^{t+H-1} \left| z_\tau^{\text{plan}} - z_{\text{goal}}^{\text{plan}} \right|,
]
subject to latent dynamics:

* (z_{\tau+1}^{\text{dyn}} = f(z_{\tau}^{\text{dyn}}, a_\tau)),
* (z_{\tau}^{\text{obj}k}) held fixed during imagined rollout (in the current implementation).

Note: in real environment, object states change (door opens, key disappears), but planning uses a “locally fixed object state” approximation and replans frequently.

---

## 3. Theoretical Motivation

### 3.1 Factorization of Latent Variables

Informally, the true state (s_t) in DoorKey can be decomposed into factors:

* (s_t = (s_t^{\text{agent}}, s_t^{\text{door}}, s_t^{\text{key}}, s_t^{\text{goal}}, s^{\text{layout}})),

  * (s^{\text{layout}}): static over episode
  * (s_t^{\text{door}}, s_t^{\text{key}}, s_t^{\text{goal}}): object-centric factors
  * (s_t^{\text{agent}}): dynamic agent state

Our latent decomposition is a *representation-theoretic* analogue:

* (z_t^{\text{dyn}} \approx \phi(s_t^{\text{agent}}))
* (z^{\text{stat}} \approx \psi(s^{\text{layout}}))
* (z_t^{\text{obj}k} \approx \xi_k(s_t^{\text{door}}, s_t^{\text{key}}, s_t^{\text{goal}}))

The **invariance loss** on (z^{\text{stat}}) enforces:
[
z^{\text{stat}}(o_t) \approx z^{\text{stat}}(o_{t'}) \quad \forall t, t' \text{ (same episode)},
]
which encourages it to encode exactly the **episode-level constant** part of the environment (layout), and *not* the changing door/key/goal states.

The **reconstruction loss** forces ((z^{\text{stat}}, z^{\text{obj}1:3})) to be sufficient for reconstructing (o_t):

[
p(o_t \mid z^{\text{stat}}, z^{\text{obj}1:3}) \approx \delta\big(o_t - D(z^{\text{stat}}, z^{\text{obj}1:3})\big),
]
so any variation in the image (e.g. key picked up, door opened) must be encoded in (z^{\text{obj}k}), not in (z^{\text{stat}}).

Meanwhile, the **dynamics prediction** exclusively uses (z^{\text{dyn}}):
[
p(z^{\text{dyn}}_{t+1} \mid z^{\text{dyn}}_t, a_t),
]
making it a minimal **control-relevant sufficient statistic** for the agent’s motion.

This matches the classical control idea that I want a representation that is:

* **Markovian** for dynamics (predictive),
* **Minimal** (no irrelevant information), and
* **Structured** into invariant and variant components.

### 3.2 Why This Helps Planning

Planning in the vanilla model operates in an entangled latent space:

[
z_t = g(o_t), \quad \hat{z}_{t+1} = f(z_t, a_t),
]
where (z_t) mixes agent state, door state, etc. The geometry of this space may be “twisted”, making straight-line interpolation between start and goal latents suboptimal or misleading.

In the disentangled model, the distance in (z^{\text{plan}})-space more faithfully splits into:

[
\left| z_t^{\text{plan}} - z_{\text{goal}}^{\text{plan}} \right|
\approx
\left| z_t^{\text{dyn}} - z_{\text{goal}}^{\text{dyn}} \right|

* \sum_k \left| z_t^{\text{obj}k} - z_{\text{goal}}^{\text{obj}k} \right|.
  ]

If the object slots indeed specialize (e.g. one for door, one for key, one for goal), then:

* Approaching the goal changes mostly the “goal slot” and agent position,
* Opening the door changes mostly the “door slot” and agent position near the door,
* Picking the key changes mostly the “key slot”.

This makes **gradient-free search** (like CEM) significantly easier: it optimizes over a latent space where different task-relevant factors live in mostly separate subspaces.

Empirically, this yields:

* Higher success rate,
* Fewer steps to goal,
* Better generalization to new layouts (e.g. Empty-5x5),
  compared to a single undifferentiated latent.

---

## 4. Hyperparameters and Latent Sizes

Given a vanilla model with single latent (z \in \mathbb{R}^{128}), a reasonable disentangled configuration that preserves similar capacity is:

* (d_{\text{dyn}} = 64),
* (d_{\text{stat}} = 32),
* (d_{\text{obj}} = 16) per object slot (3 slots → 48 dims).

Total encoder output dims ≈ (64 + 32 + 48 = 144), similar to 128 but structured.

Loss weights (good practical starting point):

* VICReg:
  (\lambda_{\text{sim}} = 1.0,\ \lambda_{\text{var}} = 1.0,\ \lambda_{\text{cov}} \approx 0.01),
* Static invariance: (\lambda_{\text{inv}} \approx 0.1),
* Reconstruction: (\lambda_{\text{rec}} \approx 1.0).

Training for at least as many epochs as the vanilla model is important (e.g. 100 vs 30), since the disentangled model is slightly more complex.

---

## 5. Further Ways to Improve Performance

Considering the current disentangled model is already outperforming the vanilla baseline, here are some directions to push it further.

### 5.1 Model Object Dynamics Explicitly

Right now, in planning, I treat (z_t^{\text{obj}k}) as constant over the imagined horizon. A more accurate model adds an “object dynamics” head:

[
\hat{z}*{t+1}^{\text{obj}k} = g_k(z_t^{\text{dyn}}, z_t^{\text{obj}1:3}, a_t),
]
and a loss:
[
\mathcal{L}*{\text{obj-dyn}} = \sum_k \left| \hat{z}*{t+1}^{\text{obj}k} - z*{t+1}^{\text{obj}k} \right|^2.
]

Then, during planning, I roll both (z^{\text{dyn}}) and (z^{\text{obj}k}) forward. This lets the model “imagine” picking up the key, opening the door, etc., entirely in latent space.

### 5.2 Multi-step Prediction Loss

Instead of only matching one-step latent targets, I can unroll for (K) steps in latent space and match the encoder’s latents at future times:

[
\mathcal{L}*{\text{multi-dyn}} = \sum*{j=1}^K \gamma^{j-1}
\left| \hat{z}*{t+j}^{\text{dyn}} - z*{t+j}^{\text{dyn}} \right|^2.
]

This encourages **long-horizon consistency**, directly improving planning stability.

### 5.3 Stronger Reconstruction

Pixel MSE is simple but crude. Two upgrades:

* **Perceptual loss** in a feature space (e.g. ConvNet features (\phi(\cdot))):
  [
  \mathcal{L}_{\text{rec}}^{\text{perc}} = |\phi(\hat{o}_t) - \phi(o_t)|^2,
  ]
* Or **cross-entropy over discrete tiles** if I convert images into one-hot tile maps.

This can sharpen how well object slots reflect semantic entities (door vs key vs goal).

### 5.4 Better Architectures for Disentanglement

I already use multiple object heads. Two natural extensions:

* **Slot attention / MONet-style attention:** enforce soft spatial assignments per slot.
* **Bottlenecked object latents** (very low dimensional (d_{\text{obj}}=4–8)) with regularizers to encourage independence (e.g. covariance penalties per slot).

Both encourage clear, object-like semantics in each slot.

### 5.5 Planning Improvements

Even with the same model, planning can be improved:

* **CEM ensembles:** sample multiple model rollouts from slightly perturbed parameters (model uncertainty) and optimize “robust cost”.
* **Value function over latents:** train a critic (V(z^{\text{plan}})) and use it as a terminal value estimate instead of pure goal distance; this is closer to Dreamer-style planning.

### 5.6 Training Tricks

Small but practical:

* Train disentangled model at least as long as the baseline (e.g. 100 epochs).
* Use cosine LR schedule or warmup.
* Add light data augmentation (color jitter, small shifts) to help invariance/generalization.

---

## 6. Summary

Compared to a **vanilla single-latent world model**, the **disentangled latent world model**:

* Splits the latent into **dynamic**, **static**, and **object-centric** components.
* Uses **invariance** on static latents and **reconstruction** to learn self-supervised object slots.
* Uses **VICReg**-style losses on a dedicated **control-relevant** dynamic latent.
* Plans in a **structured latent space** where agent, door, key, and goal information are geometrically more separable.

