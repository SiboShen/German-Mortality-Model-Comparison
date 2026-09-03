# German-Mortality-Model-Comparison

This repository contains the R and Python code used in the thesis
"Stochastic Mortality Modelling under a Binomial Framework: A Comparative Study of German Data".

## Models

- M1 and M2: Lee–Carter-type models
- M5 and M6: Cairns–Blake–Dowd-type models
- two-parameter and three- parameter: Aro & Pennanen models

## Software

- R 
- Python

## Data

German mortality data were obtained from the Human Mortality Database.
The raw data are not redistributed in this repository because they are
subject to the HMD terms of use.

## Reproduction order

1. Prepare the HMD data files.
2. Run the R scripts for M1, M2, M5 and M6.
3. Run the Python scripts for Aro & Pennanen models.
4. Run the out-of-sample and projection scripts.

## Output

The scripts reproduce the model estimates, AIC/BIC comparisons,
residual plots, out-of-sample error tables and long-term projections
reported in the thesis.
