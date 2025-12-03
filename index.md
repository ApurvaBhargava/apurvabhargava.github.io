---
layout: default
title: "Home"
---

<section class="hero">
  <div>
    <div class="hero-highlight">
      <span>Apurva Bhargava</span>
      <span>· AI Research & ML Engineering</span>
    </div>
    <h1 class="hero-main-title">
      Apurva Bhargava
    </h1>
    <p class="hero-subtitle">
      I work at the intersection of <strong>energy-based models</strong>, <strong>JEPAs</strong>,
      and <strong>model predictive control</strong>, with a focus on controllable reasoning, disentangled
      latent spaces, and practical ML systems in production.
    </p>

    <div class="hero-pill-row">
      <span class="pill">Energy-Based Models</span>
      <span class="pill">JEPA / World Models</span>
      <span class="pill">System‑2 Reasoning</span>
      <span class="pill">MPC &amp; Planning</span>
      <span class="pill">Document Intelligence</span>
    </div>

    <div class="hero-actions">
      <a class="btn btn-primary" href="/pdf/Apurva_Bhargava_CV.pdf" target="_blank" rel="noopener">
        View CV
      </a>
      <a class="btn btn-ghost" href="/research">
        Research interests
      </a>
      <a class="btn btn-ghost" href="/projects">
        Selected projects
      </a>
    </div>
  </div>

  <div class="hero-profile">
    <div class="profile-frame">
      <img src="/assets/img/profile.jpeg" alt="Profile photo of Apurva (replace this image file with your own).">
    </div>
    <div class="hero-meta">
      <span>Machine Learning Engineer · Document Intelligence &amp; Fraud Detection</span>
      <span>MS in Data Science (NYU) · MSCS (Westcliff, in progress)</span>
    </div>
  </div>
</section>

<section class="section">
  <div class="section-header">
    <h2 class="section-title">Currently</h2>
    <p class="section-subtitle">What I&apos;m focused on right now</p>
  </div>
  <div class="grid">
    <article class="card">
      <h3 class="card-title">Reasoning‑centric models</h3>
      <p class="card-body">
        Exploring architectures that separate <em>world models</em> from <em>configurators</em>,
        drawing on EBMs and JEPAs to capture structure while retaining control over outputs.
      </p>
      <div class="chip-row">
        <span class="chip">JEPA</span>
        <span class="chip">Latent planning</span>
        <span class="chip">Disentanglement</span>
      </div>
    </article>

    <article class="card">
      <h3 class="card-title">Applied ML systems</h3>
      <p class="card-body">
        Building production pipelines for document intelligence and fraud detection: OCR,
        representation learning, and anomaly detection on paystubs and bank statements.
      </p>
      <div class="chip-row">
        <span class="chip">MLOps</span>
        <span class="chip">Document AI</span>
        <span class="chip">Fraud signals</span>
      </div>
    </article>

    <article class="card">
      <h3 class="card-title">PhD preparation</h3>
      <p class="card-body">
        Designing research projects that connect structured argumentation, EBMs, and planning‑style
        reasoning, with an eye towards AGI‑adjacent questions rather than narrow benchmarks.
      </p>
      <div class="chip-row">
        <span class="chip">System‑2</span>
        <span class="chip">Abstract argumentation</span>
        <span class="chip">PhD applications</span>
      </div>
    </article>
  </div>
</section>

<section class="section">
  <div class="section-header">
    <h2 class="section-title">Highlights</h2>
    <p class="section-subtitle">A few representative projects</p>
  </div>
  <div class="grid">
    <article class="card">
      <h3 class="card-title">Energy‑Based Abstract Argumentation</h3>
      <p class="card-meta">Research prototype</p>
      <p class="card-body">
        Modeling argument acceptability with an energy function over attack/defense graphs,
        enabling graded, noise‑tolerant reasoning instead of brittle symbolic semantics.
      </p>
      <div class="chip-row">
        <span class="chip">EBM</span>
        <span class="chip">Argumentation</span>
      </div>
    </article>

    <article class="card">
      <h3 class="card-title">JEPA‑style latent world model</h3>
      <p class="card-meta">Planning in grid‑world environments</p>
      <p class="card-body">
        Early experiments on factored latent spaces for controllable rollouts, connecting
        JEPA‑style training with MPC‑like control over future states.
      </p>
      <div class="chip-row">
        <span class="chip">JEPA</span>
        <span class="chip">MPC</span>
      </div>
    </article>

    <article class="card">
      <h3 class="card-title">Document fraud detection</h3>
      <p class="card-meta">Industry · In production</p>
      <p class="card-body">
        Siamese embeddings over “good” vs “suspicious” fonts and layouts to flag anomalous
        paystubs, complementing rule‑based checks with learned similarity structure.
      </p>
      <div class="chip-row">
        <span class="chip">Siamese nets</span>
        <span class="chip">Anomaly detection</span>
      </div>
    </article>
  </div>
</section>
