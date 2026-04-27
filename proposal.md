# Inference-Time Safety Guidance for Diffusion Models

Boyu Yang · Luojia Xia · William Zhang · Vincent Lin

---

# 1. Problem Statement and Motivation

Modern text-to-image diffusion models can generate highly realistic images from natural language prompts, but they may also produce unsafe visual content (e.g., violence, weapons, explicit material) even under seemingly benign or adversarial prompts.

Existing safety mechanisms fall into three broad families:

- **Prompt-level defenses** (prompt rewriting, negative prompts): cheap but easily bypassed by jailbreak prompts.
- **Post-hoc output filtering**: rejects unsafe samples after the full generation, wasting compute and producing no usable output.
- **Model-level concept erasure** (e.g., ESD, fine-tuning): requires retraining and permanently alters the model.
- **Inference-time concept guidance** (e.g., Safe Latent Diffusion, SLD): steers sampling away from unsafe text embeddings without retraining.

In this project we focus on the fourth family. Rather than steering with a *text-embedding direction* as SLD does, we treat safety as an **explicit energy term defined by a pretrained safety classifier** and inject its gradient into the sampling trajectory via the predicted clean image $\hat{x}_0(z_t)$. This reframes safety as a form of *classifier guidance* on the denoising ODE/SDE, which (i) decouples safety from any single textual description of the unsafe concept, (ii) allows multiple classifiers to be composed, and (iii) exposes a principled knob — the guidance strength $\lambda$ — for studying the safety/quality/efficiency tradeoff.

The central research questions are:

1. How does **classifier-energy guidance** compare to **text-embedding guidance (SLD)** and **negative prompts** in terms of safety suppression, prompt fidelity, and robustness to adversarial (jailbreak) prompts?
2. Can a lightweight **adaptive $\lambda$ schedule** — driven by the online safety-score trajectory — remove the need for per-prompt tuning?
3. Does **multi-classifier composition** (violence + NSFW + weapons) interfere constructively or destructively across hazard categories?

---

# 2. Dataset and Evaluation Prompts

We do not retrain the diffusion model. We evaluate on standardized risky-prompt benchmarks with pretrained safety scorers.

Prompt sources:

- **T2I-RiskyPrompt**: https://github.com/datar001/T2I-RiskyPrompt
- **MLCommons AILuminate**: https://github.com/mlcommons/ailuminate
- **UnsafeBench**: https://huggingface.co/datasets/yiting/UnsafeBench
- **Adversarial / jailbreak prompts**: a small slice from **MMA-Diffusion** and/or **Ring-A-Bell** to stress-test robustness against prompts that evade text-level filters.

We curate a stratified evaluation set of ~800–1,000 prompts across:

- Violence / Gore
- Weapons
- Hate / Harassment
- Explicit Content
- Borderline prompts
- Neutral baseline prompts (for false-positive measurement)
- Adversarial / jailbreak prompts (robustness slice)

Safety scorers (also used as guidance signals where differentiable):

- **CLIP zero-shot concept scoring** for dangerous concepts (explosion, weapon, blood, nudity, …).
- **Q16** and/or **NudeNet / NSFW classifiers** as independent evaluators to avoid circular evaluation when CLIP is used inside the guidance loop.

To prevent circularity, the **classifier used for guidance is disjoint from the classifier used for final evaluation**.

---

# 3. Proposed Deep Learning Approach

**Backbone.** Stable Diffusion v1.5 (frozen).

**Safety energy.** Let $c_\phi(\cdot)$ be a pretrained safety classifier returning a score in $[0,1]$ for an unsafe concept. Define the hinge energy

$$
L_{\text{unsafe}}(x)=\max\bigl(0,\; c_\phi(x)-\tau\bigr),
$$

with threshold $\tau$. For multi-concept composition,

$$
L_{\text{unsafe}}(x)=\sum_k w_k \max\bigl(0,\; c_\phi^{(k)}(x)-\tau_k\bigr).
$$

**Classifier guidance on the predicted clean image.** A naïve $\nabla_{z_t} L_{\text{unsafe}}(\mathcal{D}(z_t))$ is noisy because $z_t$ is far from the data manifold at large $t$. Following *Universal Guidance* (Bansal et al., 2023), we use Tweedie's formula to estimate the clean latent

$$
\hat{z}_0(z_t) \;=\; \frac{z_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(z_t,c)}{\sqrt{\bar\alpha_t}},
\qquad
\hat{x}_0 \;=\; \mathcal{D}(\hat{z}_0),
$$

compute the safety gradient on $\hat{x}_0$, and inject it into the DDIM update:

$$
z_{t-1} \;=\; \text{DDIM}(z_t,\epsilon_\theta(z_t,c))
\;-\; \lambda_t \,\nabla_{z_t} L_{\text{unsafe}}\bigl(\hat{x}_0(z_t)\bigr).
$$

This fixes the two issues of a latent-space gradient: (i) the safety signal is evaluated on an actual image rather than a noisy latent, and (ii) the update remains a proper DDIM step plus a guidance correction.

**Adaptive $\lambda$ schedule.** Instead of hand-tuning per prompt, we set

$$
\lambda_t \;=\; \lambda_0 \cdot \sigma\bigl(\beta\,(c_\phi(\hat{x}_0)-\tau)\bigr)\cdot \mathbf{1}[t\in[t_{\min}, t_{\max}]],
$$

i.e., guidance strength is modulated by the *current* violation margin and restricted to a timestep window (empirically, mid-to-late steps where $\hat{x}_0$ is informative). Only $\lambda_0,\beta,\tau$ need to be tuned, globally.

**Efficiency tricks.**

- Compute the safety gradient on a **downsampled** $\hat{x}_0$ to cut decoder/classifier cost.
- Update every $k$ steps (skip-guidance) in the window — a cheap knob for the quality/latency tradeoff.

**Methods compared.**

1. Vanilla Stable Diffusion.
2. Negative-prompt baseline.
3. **Safe Latent Diffusion (SLD)** — text-embedding-direction guidance.
4. Output rejection sampling (post-hoc filter).
5. **Ours: classifier-energy guidance on $\hat{x}_0$** with adaptive $\lambda$.
6. **Ours + multi-classifier composition.**

Baselines 3 and 4 are the critical comparisons for positioning our contribution.

---

# 4. Anticipated Challenges

## Noisy / Unstable Guidance Gradients

**Challenge.** Gradients through VAE + CLIP at noisy latents can be high-variance.
**Mitigation.** Use $\hat{x}_0$ parameterization (above); gradient clipping; restrict guidance to a mid-timestep window; skip-guidance every $k$ steps.

## Safety / Quality Tradeoff

**Challenge.** Strong guidance pushes samples off-manifold, degrading realism and CLIP alignment.
**Mitigation.** Adaptive $\lambda$ driven by the violation margin; joint reporting of safety and CLIP-score Pareto curves rather than a single operating point.

## Computational Overhead

**Challenge.** Per-step backprop through VAE decoder + safety classifier is expensive.
**Mitigation.** Downsample $\hat{x}_0$ before scoring; guidance window instead of every step; skip-guidance; cache $\epsilon_\theta$ call already required by DDIM.

## False Positives on Benign Prompts

**Challenge.** Safety classifiers over-fire on neutral content.
**Mitigation.** Hinge with calibrated $\tau$; dedicated neutral-prompt slice to measure FPR; report FPR alongside unsafe-rate.

## Cross-Category Generalization

**Challenge.** A single classifier may be effective for NSFW but weak for weapons/hate symbols.
**Mitigation.** Multi-classifier composition with per-category $w_k,\tau_k$; report category-wise metrics.

## Evaluation Circularity

**Challenge.** Using the same CLIP for both guidance and evaluation inflates reported safety gains.
**Mitigation.** Guidance classifier and evaluation classifier are **disjoint** (e.g., guide with CLIP, evaluate with Q16 + NudeNet, or vice versa).

---

# 5. Evaluation Strategy

## Qualitative

Side-by-side generations for representative prompts across all six methods; failure-mode gallery (quality collapse, over-suppression, residual unsafe content).

## Quantitative

**Safety (disjoint evaluator).**
- Unsafe rate (%) above threshold.
- Mean safety score; score-distribution shift vs. vanilla.
- **Jailbreak robustness**: unsafe-rate on the MMA-Diffusion / Ring-A-Bell slice.

**Quality / fidelity.**
- CLIP text-image similarity on benign prompts.
- FID on a neutral subset (small-scale).
- Small blind human preference study (~100 pairs).

**Efficiency.**
- Wall-clock seconds/sample; peak VRAM.
- Overhead vs. vanilla and vs. SLD.

**Tradeoff analysis.**
- Pareto curve of unsafe-rate vs. CLIP-score as $\lambda_0$ sweeps.
- Category-wise breakdown for the multi-classifier variant.
- Ablation: fixed $\lambda$ vs. adaptive $\lambda$; full-resolution vs. downsampled safety gradient; guidance window vs. all-steps.

---

# References

1. Schramowski et al. *Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models.* CVPR 2023.
2. Bansal et al. *Universal Guidance for Diffusion Models.* CVPR 2023 Workshop.
3. Kim et al. *Training-free Safe Denoisers for Safe Use of Diffusion Models.* arXiv:2502.08011, 2025.
4. Yang et al. *MMA-Diffusion: Multimodal Attack on Diffusion Models.* CVPR 2024.
5. Tsai et al. *Ring-A-Bell! How Reliable are Concept Removal Methods for Diffusion Models?* ICLR 2024.
6. Schramowski et al. *Q16: Can Machines Help Us Answering Question 16 in Datasheets?* 2022.
7. Gandikota et al. *Erasing Concepts from Diffusion Models (ESD).* ICCV 2023.
