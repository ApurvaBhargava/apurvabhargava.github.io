---
layout: page
title: "Research interests"
subtitle: "System‑2 reasoning, structured energy landscapes, and controllable latent spaces."
---

## Core themes

1. **Energy‑based models for structured reasoning**  
   Using EBMs to score configurations of arguments, states, or plans, rather than just individual predictions. This
   offers a way to combine discrete structure (graphs, constraints) with continuous optimization.

2. **Joint embedding predictive architectures (JEPA) & world models**  
   Learning latent spaces that encode <em>what matters</em> for future prediction, without being forced to model every
   pixel or token. I am particularly drawn to JEPAs that:
   - Factorize latent spaces into meaningful components  
   - Use constraints or regularization to encourage disentanglement  
   - Support planning‑style rollouts instead of just one‑step prediction  

3. **Model predictive control (MPC) in latent space**  
   Using learned models not just to predict, but to <em>search</em> over possible futures and choose trajectories that
   optimize a task. I am interested in:
   - MPC‑style inner loops over latent states  
   - Explicit trade‑offs between exploration, robustness, and control  
   - Interfaces between symbolic constraints and continuous optimization  

---

## Example directions I am exploring

- **Energy‑Based Abstract Argumentation**  
  Representing argumentation frameworks as energy landscapes, where low‑energy configurations correspond to coherent
  sets of accepted arguments, and gradient‑based methods can explore near‑by alternatives.

- **Latent‑space control in grid‑worlds and toy environments**  
  Using JEPAs to learn compact world models, then layering an MPC‑like planner on top, before scaling to richer domains.

- **Reasoning traces vs. justified outputs in LLMs**  
  Distinguishing between a model’s internal reasoning and its external explanation, and studying when structured
  mechanisms (EBMs, JEPA‑like modules) actually improve robustness over prompt‑based chain‑of‑thought alone.

If you work on related topics and would like to discuss ideas or potential collaborations, feel free to
<a href="/contact">reach out</a>.
