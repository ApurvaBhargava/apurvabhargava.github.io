# Latent-space Planning and Disentangled Control in MiniGrid

This project explores how far we can go with a Predictive Latent Dynamics Model (PLDM) trained purely from trajectories collected in MiniGrid.

## Problem Setting

- **Environment**: MiniGrid DoorKey 5x5 (plus transfer to Empty / Dynamic-Obstacles variants). The agent must pick up a key, unlock a door, and reach the goal.
- **Data**: ~1 200 random-play trajectories (24 576 subsequences of length 8 after windowing). Observations are 64×64 RGB arrays, actions are discrete (turn left/right, move, pickup, toggle, done).
- **Goal**: Learn a latent world model that supports planning via a Cross-Entropy Method (CEM) optimiser without hand-coded state machines.

## Model Intuition

| Component | Clean PLDM | Disentangled PLDM |
|-----------|------------|-------------------|
| Encoder   | CNN → 128-d latent `z` | CNN → `(z_dyn, z_stat)` (each 64-d) |
| Dynamics  | Residual MLP conditioned on actions | Same, but only predicts `z_dyn` (agent-controllable factors) |
| Planner   | CEM in latent space; minimise `‖z_t − z_goal‖` | Same, still acting only in `z_dyn` space |

The disentangled variant aims to keep layout/door configuration in a static head (`z_stat`) while leaving agent-controllable variations (`z_dyn`) to the predictor and planner.

## Objective (Mathematics)

Both notebooks optimise a VICReg-style loss on consecutive latent pairs:

```
ℒ_VICReg = ℒ_sim(z_pred, z_target)
         + λ_std ∑_i max(0, γ - σ(z[:, i]))
         + λ_cov ∑_{i≠j} Cov(z[:, i], z[:, j])²
```

where:
- `ℒ_sim` is MSE between predicted and target latents,
- `σ` denotes per-dimension batch std (kept above γ = 1),
- the covariance penalty discourages collapse across coordinates.

The disentangled model adds an *invariance* loss:

```
ℒ_inv = ‖z_stat(t) − z_stat(t+1)‖²
```

and trains with `ℒ_total = ℒ_VICReg + λ_inv ℒ_inv` (with `λ_inv = 0.04` in the notebook).

## Training Procedure

1. Generate trajectories with random policies to cover the maze.
2. Window each trajectory into overlapping sequences of length 8 to form (observations, actions, next_observations).
3. Optimise PLDM with Adam (`3e-4`), gradient clipping, and tracking of train/val splits.
4. Save the best checkpoint by validation loss; log similarity/std/cov components for diagnostics.
5. Visualise latent behaviour (variance over time, PCA, etc.) to ensure the representation moves when the agent moves.

## Planning Procedure

1. Encode current observation and goal observation into latent space.
2. Use a CEM planner (horizon 15, 500 samples, 50 elites) that:
   - Samples candidate action sequences.
   - Rolls them out through `predict_step` (which propagates only `z_dyn`).
   - Scores sequences via latent distance to the goal (`‖z_dyn - z_dyn_goal‖`).
   - Refits the action distribution to the elites and repeats.
3. Execute the best action prefix in MiniGrid; optionally replan every few steps for robustness.

## Results Snapshot

| Scenario | Clean PLDM | Disentangled PLDM |
|----------|------------|-------------------|
| DoorKey-5x5 (training task) | High success (reaching goal in ≤15 steps with replanning) | Suffers late failures; often stalls near the door/goal despite low training loss |
| Empty/Dynamic transfer | Successfully navigates in 3–10 steps | Finishes trivial mazes but becomes unstable once door/key logic is required |
| Covariance regulariser | Stays low / well-conditioned | Blows up to ~0.32 despite training success message, signalling correlated latents |

The clean model reliably solves DoorKey after ~15 epochs, while the disentangled model, despite lower MSE, frequently fails 2–3 steps short of the target.

## Diagnosis Highlights

- The invariance loss forces `z_stat` to remain constant even though the world *does* change (door unlocks, key disappears), leading to conflicting gradients near the goal.
- The planner/dynamics ignore `z_stat`, so splitting the latent effectively removed half of the information available to the optimiser.
- VICReg covariance loss increases in the disentangled run, hinting that the halved `z_dyn` capacity plus the invariance penalty induce highly correlated features – planners rely on brittle signals.


## Suggested Next Steps

1. **Gate the invariance penalty** to frames where the layout truly stays constant (no key pickup / door toggling) or anneal `λ_inv` over training.
2. **Condition planning on both heads** (`[z_dyn, z_stat]`) or feed `z_stat` as context so the optimiser “knows” about static objects.
3. **Shared capacity checks**: widen `z_dyn` (e.g., 96/32 split) or allow limited bleed-over via adversarial penalties instead of hard MSE.
4. **Better positives for invariance**: compare across trajectories that share the same seed/layout instead of consecutive timesteps.

With these adjustments, the disentangled model should be able to turn its cleaner representation into tangible planning gains.