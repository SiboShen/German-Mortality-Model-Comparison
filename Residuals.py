#!/usr/bin/env python3
"""Create standardized binomial deviance-residual plots for AP2 and AP3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("/Users/sibo/Desktop/Data")
YEARS = np.arange(1956, 2021)
AGES = np.arange(18, 101)
DEATH_EPSILON = 1e-5
FIGURE_WIDTH = 9.0
FIGURE_HEIGHT = 5.5


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def read_age_year_matrix(filename):
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


def standardized_binomial_deviance_residuals(
    initial_exposure,
    deaths,
    fitted_survival_probability,
    number_of_parameters,
):
    exposure = initial_exposure.to_numpy(dtype=float)
    death_count = deaths.to_numpy(dtype=float) + DEATH_EPSILON
    fitted_p = fitted_survival_probability.to_numpy(dtype=float)

    if exposure.shape != death_count.shape or exposure.shape != fitted_p.shape:
        raise ValueError("Exposure, death and probability matrices must have equal shapes.")
    if np.any(exposure <= 0):
        raise ValueError("Initial exposures must be positive.")

    observed_q = death_count / exposure
    fitted_q = np.clip(1.0 - fitted_p, 1e-12, 1.0 - 1e-12)

    if np.any(observed_q <= 0) or np.any(observed_q >= 1):
        raise ValueError("Observed death probabilities must lie inside (0, 1).")

    deviance = 2.0 * exposure * (
        observed_q * np.log(observed_q / fitted_q)
        + (1.0 - observed_q)
        * np.log((1.0 - observed_q) / (1.0 - fitted_q))
    )

    observations = int(exposure.size)
    degrees_of_freedom = observations - int(number_of_parameters)
    if degrees_of_freedom <= 0:
        raise ValueError("Residual degrees of freedom must be positive.")

    dispersion = float(np.sum(deviance) / degrees_of_freedom)
    if not np.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("The estimated dispersion parameter is not positive.")

    sign = np.sign(observed_q - fitted_q)
    residual_values = sign * np.sqrt(np.abs(deviance) / dispersion)

    if not np.all(np.isfinite(residual_values)):
        raise ValueError("Residual calculation produced non-finite values.")

    return pd.DataFrame(
        residual_values,
        index=initial_exposure.index,
        columns=initial_exposure.columns,
    )


def residuals_to_long(residuals):
    result = (
        residuals.rename_axis(index="Calendar year", columns="Age")
        .stack()
        .rename("Residual")
        .reset_index()
    )
    result["Year of birth"] = result["Calendar year"] - result["Age"]
    return result


def plot_residuals(
    male_residuals,
    female_residuals,
    output_filename,
    residual_ylim,
):
    male = residuals_to_long(male_residuals)
    female = residuals_to_long(female_residuals)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        sharey=True,
    )
    plot_specs = (
        ("Age", "Age"),
        ("Calendar year", "Calendar year"),
        ("Year of birth", "Year of birth"),
    )

    for column, (variable, label) in enumerate(plot_specs):
        axes[0, column].scatter(
            male[variable],
            male["Residual"],
            color="blue",
            s=5,
            alpha=0.75,
            linewidths=0,
        )
        axes[1, column].scatter(
            female[variable],
            female["Residual"],
            color="red",
            s=5,
            alpha=0.75,
            linewidths=0,
        )
        axes[0, column].set_xlabel(label)
        axes[1, column].set_xlabel(label)

    axes[0, 0].set_ylabel("Residuals (Male)")
    axes[1, 0].set_ylabel("Residuals (Female)")

    for axis in axes.flat:
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_ylim(residual_ylim)
        axis.grid(False)

    fig.subplots_adjust(
        hspace=0.34,
        wspace=0.18,
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=0.98,
    )
    # Do not use bbox_inches="tight": it changes the PDF media-box size and
    # makes the AP figures inconsistent with the 9 x 5.5 inch StMoMo PDFs.
    fig.savefig(DATA_DIR / output_filename, format="pdf")
    plt.close(fig)


def main():
    exposure = {
        "Male": read_age_year_matrix("E_male_18.csv"),
        "Female": read_age_year_matrix("E_female_18.csv"),
    }
    deaths = {
        "Male": read_age_year_matrix("D_male_18.csv"),
        "Female": read_age_year_matrix("D_female_18.csv"),
    }

    residuals_by_model = {}

    for factors in (2, 3):
        number_of_parameters = factors * len(YEARS)
        residuals = {}

        for sex in ("Male", "Female"):
            fitted_p = read_survival_probabilities(
                f"{sex.lower()}_prob_18_{factors}.csv"
            )
            residuals[sex] = standardized_binomial_deviance_residuals(
                exposure[sex],
                deaths[sex],
                fitted_p,
                number_of_parameters,
            )

        residuals_by_model[factors] = residuals

    all_residual_values = np.concatenate(
        [
            residuals_by_model[factors][sex].to_numpy().ravel()
            for factors in (2, 3)
            for sex in ("Male", "Female")
        ]
    )
    all_residual_values = all_residual_values[np.isfinite(all_residual_values)]
    if all_residual_values.size == 0:
        raise ValueError("No finite residual values were found.")

    residual_limit = max(
        3.5,
        np.ceil(2.0 * np.max(np.abs(all_residual_values))) / 2.0,
    )
    shared_ylim = (-residual_limit, residual_limit)

    for factors in (2, 3):
        residuals = residuals_by_model[factors]
        plot_residuals(
            residuals["Male"],
            residuals["Female"],
            f"residuals_AP{factors}.pdf",
            shared_ylim,
        )


if __name__ == "__main__":
    main()

