# Manifold-Anchored Dynamics for Few-Shot Class-Incremental Learning

This repository contains the official implementation of:

**Manifold-Anchored Dynamics: Harmonizing Stability and Plasticity for Few-Shot Class-Incremental Learning**

## Introduction

Few-Shot Class-Incremental Learning (FSCIL) aims to continuously learn novel classes from only a few labeled samples while preserving the knowledge of previously learned classes.

In this work, we propose **Manifold-Anchored Dynamics (MAD)**, a representation-reuse framework built on a frozen CLIP-ViT backbone. MAD reduces unstable incremental adaptation and improves classifier expansion through two key components:

- **Manifold Stability Regularization (MSR)**: uses normalized base-class anchors and a small replay memory to reduce base-class representation drift.
- **Non-parametric Manifold Calibration (NMC)**: constructs novel-class classifier weights from normalized support features without iterative classifier optimization.

## Framework

MAD follows a three-stage pipeline:

1. **Base-session Training**  
   Train the task-specific MAB and classifier with a frozen CLIP-ViT backbone.

2. **MSR-based Incremental Adaptation**  
   Adapt the task-specific module using novel-class support samples while applying MSR on replayed base samples.

3. **NMC-based Classifier Expansion**  
   Construct novel-class classifier weights non-parametrically and expand the global classifier.

## Requirements

The code is implemented with Python and PyTorch.

```bash
pip install -r requirements.txt
