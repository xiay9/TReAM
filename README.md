We will release the code when the paper is accepted.

# TReAM: Temporal Recency Alignment Across Modules for Multimodal Emotion-Cause Pair Extraction

This repository will contain the official implementation of our paper:

**Temporal Recency Alignment Across Modules for Multimodal Emotion–Cause Pair Extraction**

TReAM is a lightweight framework for **Multimodal Emotion–Cause Pair Extraction (MECPE)**. It addresses temporal inconsistency across modular MECPE systems by learning a shared dialogue-paced temporal scale and injecting it into both global pair reasoning and local speaker interaction modules.

## Overview

Multimodal Emotion–Cause Pair Extraction aims to identify emotion utterances, cause utterances, and structured emotion-cause links in multi-speaker multimodal dialogues.

Existing modular MECPE models often combine sequential aggregation and speaker-interaction modules. However, these modules may encode different temporal recency assumptions, resulting in cross-module temporal misalignment and unstable pair prediction.

TReAM addresses this issue through two key components:

- **ATDG: Adaptive Temporal Decay Generator**  
  Infers a dialogue-level temporal scale from label-free dialogue statistics such as dialogue length, speaker distribution, speaker-switching rate, and turn-taking patterns.

- **DP: Dual-Path Temporal Injection**  
  Injects the shared temporal scale into:
  - **Kernel Smoothing (KS)** for global, order-respecting pair reasoning.
  - **Speaker Graph (SG)** for local speaker-aware utterance-level prediction.

Importantly, pair scoring is performed only through the KS path, while SG is used for utterance-level emotion and cause prediction. This design keeps structured pair reasoning temporally coherent and avoids interference from local interaction dynamics.

## Main Features

- Dialogue-adaptive temporal recency modeling.
- Cross-module temporal alignment between sequential aggregation and speaker interaction.
- Dual-path design separating pair-level reasoning and utterance-level prediction.
- Support for text-only and multimodal settings.
- Experiments on ECF and MECAD datasets.
- Robust performance on sparse and non-local emotion-cause links.

## Method

TReAM consists of three main stages:

1. **Multimodal Utterance Encoding**  
   Each utterance is represented using text, audio, and video features.

2. **Adaptive Temporal Scale Generation**  
   ATDG maps label-free dialogue statistics to a shared latent temporal factor, which is then used to generate temporal decay parameters.

3. **Dual-Path Reasoning**
   - **KS Path:** Performs temporally aligned global aggregation and serves as the only input to the pair scorer.
   - **SG Path:** Performs local cross-speaker interaction and supports utterance-level emotion and cause classification.

## Results

On the ECF dataset, TReAM achieves strong performance in the full multimodal setting:

| Task | F1 |
|---|---:|
| Emotion Extraction | 79.33 |
| Cause Extraction | 73.26 |
| Pair Extraction | 57.92 |

On the MECAD dataset, TReAM also improves pair extraction performance and achieves a Pair F1 of 52.21 in the full multimodal setting.

## Repository Structure

The code will be organized as follows after release:

```text
TReAM/
├── data/                 # Dataset files and preprocessing scripts
├── configs/              # Configuration files
├── models/               # Model implementation
├── modules/              # ATDG, KS, SG, and related components
├── scripts/              # Training and evaluation scripts
├── utils/                # Utility functions
├── results/              # Experimental logs and outputs
├── requirements.txt      # Python dependencies
└── README.md
