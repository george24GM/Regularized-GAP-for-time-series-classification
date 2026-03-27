# Regularized-GAP-for-time-series-classification


# GAP vs WGAP for Time Series Classification

This repository contains a PyTorch implementation for comparing three pooling strategies in a Fully Convolutional Network (FCN) for univariate time series classification:

- **GAP**: standard global average pooling
- **WGAP**: unconstrained weighted global average pooling
- **reg_WGAP**: weighted global average pooling with a smoothness penalty and cross-validated regularization

The code runs end-to-end experiments on multiple UCR-style datasets, evaluates each method across multiple random seeds, selects the regularization parameter for `reg_WGAP` by stratified cross-validation, and saves detailed logs and summary tables.

The FCN backbone follows the standard time-series FCN design with three 1D convolutional blocks and global pooling, a strong baseline in time series classification. The general deep-learning context for time-series classification is reviewed in detail in Ismail Fawaz et al. (2019)

---

## Features

- Pure **PyTorch** implementation
- Supports **multiple datasets** and **multiple seeds**
- Three pooling variants:
  - `GAP`
  - `WGAP`
  - `reg_WGAP`
- **Stratified K-fold cross-validation** for selecting the regularization parameter `lambda`
- Optional **per-series z-normalization**
- Automatic support for:
  - `cuda`
  - `mps`
  - `cpu`
- Logging to both:
  - terminal
  - results file
- Optional export of a CSV with selected `lambda` and corresponding test metrics

---

## Model overview

The classifier is a 1D Fully Convolutional Network with:

- Conv1dSame(1 → 128, kernel size 8)
- BatchNorm + ReLU
- Conv1dSame(128 → 256, kernel size 5)
- BatchNorm + ReLU
- Conv1dSame(256 → 128, kernel size 3)
- BatchNorm + ReLU
- Pooling layer:
  - `GAP`: mean over time
  - `WGAP`: learnable weighted sum over time
  - `reg_WGAP`: learnable weighted sum with smoothness penalty
- Final linear layer to class logits

The use of an FCN with three convolutional blocks and global pooling is standard in deep learning baselines for time series classification .

---

## Penalized weighted pooling

For `reg_WGAP`, the pooling weights \(a_1,\dots,a_T\) are learned jointly with the network and regularized through the smoothness penalty

\[
\sum_{t=1}^{T-1} (a_{t+1} - a_t)^2.
\]

This encourages smoother temporal importance profiles and avoids highly irregular weighting schemes.

The regularization strength `lambda` is selected by cross-validation on the **training set only**.

---

## Expected dataset format

The script expects a UCR-style directory structure:

```text
datasets/
├── Adiac/
│   ├── Adiac_TRAIN.txt
│   └── Adiac_TEST.txt
├── Beef/
│   ├── Beef_TRAIN.txt
│   └── Beef_TEST.txt
├── ...
