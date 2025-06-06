# Diffusion Piecewise Exponential Models

This package provides an interface for fitting Diffusion Piecewise Exponential models as introduced in [*Diffusion piecewise exponential models for survival extrapolation using Piecewise Deterministic Monte Carlo*](https://arxiv.org/abs/2505.05932).

Calling the `pem_fit` function runs two chains of the model, and outputs model parameters and MCMC diagnostics using MCMCDiagnosticTools.jl.

A simple example based on simulated data is as follows:

```


```

## Calling the package via R

Analysis for Health Technology Assessment will often use R. This package can be called via R using the JuliaCall package. 

Repeating the above example:

```


```

## Current work

This package contains all the functionality required to implement these models and generate extrapolations. Current work is focused on:

* Functions for automatically plotting outputs
* Improved documentation and examples
* Improved interoperability with R and relevant HTA focused packages 
