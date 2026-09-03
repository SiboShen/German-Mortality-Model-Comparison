#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 23:00:25 2025

@author: sibo
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln


DATA_DIR = Path("/Users/sibo/Desktop/Data")
YEARS = np.arange(1956, 2021)
AGES = np.arange(18, 101)


def read_age_year_matrix(filename):
    """Read a Year-by-Age matrix and enforce the common analysis grid."""
    data = pd.read_csv(DATA_DIR / filename)

    if "Year" in data.columns:
        data = data.set_index("Year")
    elif data.columns[0].startswith("Unnamed"):
        data = data.set_index(data.columns[0])
    else:
        raise ValueError(f"{filename} must contain a Year column.")

    data.index = pd.to_numeric(data.index, errors="raise").astype(int)
    data.columns = pd.to_numeric(data.columns, errors="raise").astype(int)
    data = data.reindex(index=YEARS, columns=AGES)

    if data.isna().any().any():
        raise ValueError(f"{filename} has missing values after Year-Age alignment.")

    return data.astype(float)


def read_survival_probabilities(filename):
    """Read the long-format fitted-survival output from A_and_P_Model.py."""
    data = pd.read_csv(DATA_DIR / filename)
    required = {"Year", "Age", "Survival Probability"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)
    data["Age"] = pd.to_numeric(data["Age"], errors="raise").astype(int)
    data["Survival Probability"] = pd.to_numeric(
        data["Survival Probability"], errors="raise"
    )

    if data.duplicated(["Year", "Age"]).any():
        raise ValueError(f"{filename} contains duplicate Year-Age rows.")

    probabilities = (
        data.pivot(index="Year", columns="Age", values="Survival Probability")
        .reindex(index=YEARS, columns=AGES)
        .astype(float)
    )

    if probabilities.isna().any().any():
        raise ValueError(f"{filename} has missing values after Year-Age alignment.")
    if ((probabilities <= 0) | (probabilities >= 1)).any().any():
        raise ValueError(f"{filename} contains probabilities outside (0, 1).")

    return probabilities


def stmomo_binomial_log_likelihood(initial_exposure, deaths, survival_probability):
    """Return the complete log-likelihood using StMoMo's convention."""
    exposure = initial_exposure.to_numpy(dtype=float)
    death_count = deaths.to_numpy(dtype=float)
    p = survival_probability.to_numpy(dtype=float)

    if exposure.shape != death_count.shape or exposure.shape != p.shape:
        raise ValueError("Exposure, death and probability matrices must have equal shapes.")
    if np.any(exposure <= 0):
        raise ValueError("Initial exposures must be positive.")
    if np.any(death_count < 0) or np.any(death_count > exposure):
        raise ValueError("Death counts must satisfy 0 <= D <= E0.")

    fitted_q = np.clip(1.0 - p, 1e-12, 1.0 - 1e-12)
    observed_q = death_count / exposure

    rounded_exposure = np.rint(exposure)
    rounded_deaths = np.rint(observed_q * exposure)
    rounded_survivors = rounded_exposure - rounded_deaths

    if np.any(rounded_survivors < 0):
        raise ValueError("Rounded deaths exceed rounded initial exposures.")

    log_combination = (
        gammaln(rounded_exposure + 1.0)
        - gammaln(rounded_deaths + 1.0)
        - gammaln(rounded_survivors + 1.0)
    )

    contributions = log_combination + exposure * (
        observed_q * np.log(fitted_q)
        + (1.0 - observed_q) * np.log1p(-fitted_q)
    )

    if not np.all(np.isfinite(contributions)):
        raise ValueError("The log-likelihood contains non-finite contributions.")

    return float(np.sum(contributions))


def information_criteria(log_likelihood, parameters, observations):
    aic = -2.0 * log_likelihood + 2.0 * parameters
    bic = -2.0 * log_likelihood + np.log(observations) * parameters
    return aic, bic


def evaluate_model(sex, model, factors, exposure, deaths, probabilities):
    log_likelihood = stmomo_binomial_log_likelihood(
        exposure, deaths, probabilities
    )
    observations = int(exposure.size)
    parameters = int(factors * len(exposure.index))
    aic, bic = information_criteria(
        log_likelihood, parameters, observations
    )

    return {
        "Model": model,
        "Sex": sex,
        "Maximum_Log_Likelihood": log_likelihood,
        "Effective_Parameters": parameters,
        "Observations": observations,
        "AIC": aic,
        "BIC": bic,
    }


def main():
    exposure = {
        "Male": read_age_year_matrix("E_male_18.csv"),
        "Female": read_age_year_matrix("E_female_18.csv"),
    }
    deaths = {
        "Male": read_age_year_matrix("D_male_18.csv"),
        "Female": read_age_year_matrix("D_female_18.csv"),
    }

    results = []
    for factors, model in ((2, "AP2"), (3, "AP3")):
        for sex in ("Male", "Female"):
            probability_file = (
                f"{sex.lower()}_prob_18_{factors}.csv"
            )
            probabilities = read_survival_probabilities(probability_file)
            results.append(
                evaluate_model(
                    sex,
                    model,
                    factors,
                    exposure[sex],
                    deaths[sex],
                    probabilities,
                )
            )

    result_table = pd.DataFrame(results)
    numeric_columns = ["Maximum_Log_Likelihood", "AIC", "BIC"]
    result_table[numeric_columns] = result_table[numeric_columns].round(2)

    ap2 = result_table.loc[result_table["Model"] == "AP2"].reset_index(drop=True)
    ap3 = result_table.loc[result_table["Model"] == "AP3"].reset_index(drop=True)

    ap2.to_csv(DATA_DIR / "goodness-of-fit-2.csv", index=False)
    ap3.to_csv(DATA_DIR / "goodness-of-fit-3.csv", index=False)

if __name__ == "__main__":
    main()
    
