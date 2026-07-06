# Hybrid Physics-Informed Framework for Black-Hole Spin Estimation

This repository contains the code and supporting material for the manuscript:

**A Hybrid Physics-Informed Framework for Black-Hole Spin Estimation**  
Stella Menziltsidou  
Department of Informatics, Democritus University of Thrace, Greece

## Overview

This project investigates a hybrid inverse Physics-Informed Neural Network (PINN) framework for black-hole spin estimation using a reduced scalar angular Teukolsky-like formulation.

The black-hole spin parameter is treated as a trainable physical quantity and optimized jointly with the neural-network approximation of the angular mode function. The training objective combines:

- physics-informed residual loss,
- boundary-condition penalties,
- angular-mode data consistency,
- normalization constraint,
- spin regularization.

The experiments are performed under controlled synthetic and Gaussian noise-contaminated angular-mode configurations.

## Important Note

The present implementation should be interpreted as a diagnostic inverse-PINN experiment rather than as a fully validated astrophysical spin-estimation pipeline.

In the supplied runs, the PINN reproduces angular-mode profiles qualitatively, but the inferred spin values cluster near the upper part of the allowed interval. This motivates further work on identifiability analysis, loss reweighting, eigenvalue inference, full Kerr perturbation modelling, and validation against high-resolution numerical solvers and astrophysical data.

## Repository Contents

```text
.
├── inverse_pinn_spin_estimation.py
├── requirements.txt
├── inverse_pinn_spin_results.rar
└── README.md
