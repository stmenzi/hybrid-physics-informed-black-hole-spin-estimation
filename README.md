# Diagnostic Physics-Informed Framework for Synthetic Black-Hole Spin Inference

This repository contains the code and supporting material for the manuscript:

**A Diagnostic Physics-Informed Framework for Synthetic Black-Hole Spin Inference from Reduced Teukolsky-Like Angular Models**  
Stella Menziltsidou  
Department of Informatics, Democritus University of Thrace, Greece

## Overview

This project investigates a diagnostic inverse Physics-Informed Neural Network (PINN) framework for synthetic black-hole spin inference using a reduced scalar angular Teukolsky-like formulation.

The black-hole spin parameter is treated as a trainable physical quantity and optimized jointly with the neural-network approximation of the angular mode function. The training objective combines:

- physics-informed residual loss,
- boundary-condition penalties,
- angular-mode data consistency,
- normalization constraint,
- spin regularization.

The experiments are performed under controlled synthetic and Gaussian noise-contaminated angular-mode configurations. The purpose of the repository is to provide a reproducible diagnostic benchmark for studying the behaviour of inverse PINN-based spin inference in a reduced angular setting.

## Important Note

The present implementation should be interpreted as a **diagnostic inverse-PINN experiment**, not as a fully validated astrophysical spin-estimation pipeline.

In the supplied runs, the PINN reproduces angular-mode profiles qualitatively, but the inferred spin values cluster near the upper part of the allowed interval rather than accurately recovering the full range of reference spins. This behaviour is interpreted as an identifiability and loss-geometry limitation of the current reduced inverse problem.

The results motivate further work on:

- identifiability analysis,
- loss reweighting,
- loss-ablation studies,
- joint eigenvalue inference,
- adaptive sampling,
- realistic uncertainty models,
- full Kerr perturbation modelling,
- validation against high-resolution numerical solvers and astrophysical data.

## Repository Contents

```text
.
├── inverse_pinn_spin_estimation.py
├── requirements.txt
├── inverse_pinn_spin_results.zip
└── README.md
