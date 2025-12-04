---
layout: page
title: "Selected projects"
subtitle: "Selected academic and independent work"
---

<div class="projects-grid">

  <!-- PROJECT 1 -->
  <div class="project-card">
    <img src="/assets/img/projects/minigrid3d.jpg" class="project-img" alt="">
    <h3 class="project-title">Latent-space Planning and Disentangled Control in MiniGrid</h3>
    <p class="project-desc">
      Extended Predictive Latent Dynamics Models (PLDM) on MiniGrid DoorKey-5×5, building a flexible RGB encoder and a receding-horizon CEM planner with trajectory and latent-space diagnostics; achieved 92–100% planning success when replanning every 6–9 steps; analyzed how replan frequency affects efficiency and reliability; prototyped a disentangled PLDM with split latent heads, action-conditioned dynamics, and invariance penalties, for cleaner latent trajectories and more interpretable planning signals.
    </p>
    <div class="project-links">
      <a href="https://github.com/ApurvaBhargava/minigrid-latent-planning-disentangled" target="_blank">GitHub →</a>
      <a href="https://apurvabhargava.github.io/writeups/minigrid-latent-planning-disentangled" target="_blank">Write-up →</a>
    </div>
  </div>

  <!-- PROJECT 2 -->
  <div class="project-card">
    <img src="/assets/img/projects/pruning.png" class="project-img" alt="">
    <h3 class="project-title">SNIP Pruning for Few-Shot Generalization and Continual Learning</h3>
    <p class="project-desc">
      Built a comprehensive empirical study of SNIP pruning for few-shot learning, transfer learning, and continual learning on CIFAR-100. The project benchmarks SNIP against dense baselines, pretrained models, EWC, and PackNet across multiple CNN architectures. Results show that SNIP improves low-data generalization and significantly mitigates catastrophic forgetting when used as an importance measure or pruning strategy.
    </p>
    <div class="project-links">
      <a href="https://github.com/ApurvaBhargava/snip-multitask-fewshot" target="_blank">GitHub →</a>
      <a href="https://apurvabhargava.github.io/writeups/snip-multitask-fewshot" target="_blank">Write-up →</a>
    </div>
  </div>

  <!-- PROJECT 3 -->
  <div class="project-card">
    <img src="/assets/img/projects/dementia.png" class="project-img" alt="">
    <h3 class="project-title">Semantic Cognition in Dense Convolutional Networks</h3>
    <p class="project-desc">
      Studied the similarity between CNN-based architectures and human biological neural systems by simulating the pattern of learning (differentiating) and forgetting (dementia) with object recognition on CIFAR-100 as the cognitive task and DenseNet-BC as the model; explored category typicality and effect of distortion using class ranking correlations.
    </p>
    <div class="project-links">
      <a href="https://github.com/ApurvaBhargava/semantic-cognition-convnets" target="_blank">GitHub →</a>
      <a href="https://github.com/ApurvaBhargava/semantic-cognition-convnets/blob/master/Project%20Paper.pdf" target="_blank">Write-up →</a>
    </div>
  </div>

  <!-- PROJECT 4 -->
  <div class="project-card">
    <img src="/assets/img/projects/optimal.png" class="project-img" alt="">
    <h3 class="project-title">Optimal Representative Training Subset Selection</h3>
    <p class="project-desc">
      Represented text documents in low-dimensional space and implemented statistical distance and sparse-coding-based methods in Python for selecting the most representative subsets, beating the active learning and topic model-based D-optimal design selection methods from literature.
    </p>
    <div class="project-links">
      <a href="https://github.com/ApurvaBhargava/OptimalSets" target="_blank">GitHub →</a>
      <a href="https://github.com/ApurvaBhargava/OptimalSets/blob/master/Project%20Paper.pdf" target="_blank">Write-up →</a>
    </div>
  </div>

  <!-- PROJECT 5 -->
  <div class="project-card">
    <img src="/assets/img/projects/gender.png" class="project-img" alt="">
    <h3 class="project-title">Gender Reinflection in Machine Translation (English to French and Spanish)</h3>
    <p class="project-desc">
      Created a novel user-aware gender reinflection + translation model that both translates and reinflects the gender as specified; also built two gendered parallel corpora (English-French and English-Spanish); the MLE and sequence-to-sequence GRU models implemented using PyTorch achieved >95% precision and >83% recall.
    </p>
    <div class="project-links">
      <a href="https://github.com/ApurvaBhargava/gender-reinflect-nlp-project" target="_blank">GitHub →</a>
      <a href="https://github.com/ApurvaBhargava/gender-reinflect-nlp-project/blob/master/NLP_Project_Report.pdf" target="_blank">Write-up →</a>
    </div>
  </div>

  <!-- PROJECT 6 -->
  <div class="project-card">
    <img src="/assets/img/projects/singan.png" class="project-img" alt="">
    <h3 class="project-title">Edge Selective Super Resolution using SinGAN</h3>
    <p class="project-desc">
      Built an MLP function approximator over SinGAN in Python to arbitrarily query a low resolution image for real-valued edge co-ordinates to perform super-resolution; this was achieved by substituting the SinGAN generators with autoencoders and feeding the encodings to an MLP to predict pixel outputs from input coordinates.
    </p>
    <div class="project-links">
      <a href="https://github.com/ApurvaBhargava/SinGAN_MLP_approximator" target="_blank">GitHub →</a>
      <a href="https://github.com/ApurvaBhargava/SinGAN_MLP_approximator/blob/master/3001ComputerVision%20Report.pdf" target="_blank">Write-up →</a>
    </div>
  </div>

</div>
