#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize
from scipy.special import expit


DATA_DIR = Path("/Users/sibo/Desktop/Data")

FULL_YEARS = np.arange(1956, 2021)
SHORT_YEARS = np.arange(1956, 2001)
AGES_18 = np.arange(18, 101)
AGES_65 = np.arange(65, 91)
RUN_ROBUSTNESS = True

KEY_COLUMNS = ["Year", "Age"]
VALUE_COLUMNS = ["Female", "Male", "Total"]


def read_mortality_file(filename):
    """Read a prepared exposure or death CSV without changing its values."""
    data = pd.read_csv(DATA_DIR / filename)

    required = set(KEY_COLUMNS + VALUE_COLUMNS)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

    data = data.loc[:, KEY_COLUMNS + VALUE_COLUMNS].copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["Age"] = pd.to_numeric(data["Age"], errors="coerce")

    for column in VALUE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=KEY_COLUMNS + VALUE_COLUMNS).copy()
    data["Year"] = data["Year"].astype(int)
    data["Age"] = data["Age"].astype(int)

    if data.duplicated(KEY_COLUMNS).any():
        duplicates = data.loc[data.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
        raise ValueError(
            f"{filename} contains duplicate Year-Age rows:\n"
            f"{duplicates.head().to_string(index=False)}"
        )

    return data.sort_values(KEY_COLUMNS).reset_index(drop=True)


def load_age_range(age_min, age_max):
    """Load central exposures and deaths for one prepared age range."""
    suffix = "18" if (age_min, age_max) == (18, 100) else "65"
    central_exposure = read_mortality_file(f"expo_total_{suffix}.csv")
    deaths = read_mortality_file(f"deat_total_{suffix}.csv")

    central_exposure = central_exposure.query(
        "1956 <= Year <= 2020 and @age_min <= Age <= @age_max"
    ).copy()
    deaths = deaths.query(
        "1956 <= Year <= 2020 and @age_min <= Age <= @age_max"
    ).copy()

    expected_rows = len(FULL_YEARS) * (age_max - age_min + 1)
    if len(central_exposure) != expected_rows or len(deaths) != expected_rows:
        raise ValueError(
            f"Age range {age_min}-{age_max} must contain {expected_rows} rows "
            "in both the exposure and death files."
        )

    exposure_keys = central_exposure.loc[:, KEY_COLUMNS]
    death_keys = deaths.loc[:, KEY_COLUMNS]
    if not exposure_keys.equals(death_keys):
        raise ValueError(
            f"Exposure and death Year-Age keys do not match for ages "
            f"{age_min}-{age_max}."
        )

    return central_exposure, deaths


def prepare_working_matrices(central_exposure, deaths, sex, years, ages):
    """Construct E0, S and D matrices for the binomial AP likelihood."""
    year_index = pd.Index(years, name="Year")
    age_columns = pd.Index(ages, name="Age")

    ec = (
        central_exposure.pivot(index="Year", columns="Age", values=sex)
        .reindex(index=year_index, columns=age_columns)
        .astype(float)
    )
    d = (
        deaths.pivot(index="Year", columns="Age", values=sex)
        .reindex(index=year_index, columns=age_columns)
        .astype(float)
    )

    if ec.isna().any().any() or d.isna().any().any():
        raise ValueError(f"Missing {sex} observations after Year-Age alignment.")
    if (ec <= 0).any().any():
        raise ValueError(f"Central exposures must be positive for {sex}.")
    if (d < 0).any().any():
        raise ValueError(f"Death counts must be non-negative for {sex}.")

    initial_exposure = ec + 0.5 * d
    survivors = initial_exposure - d

    if (survivors < 0).any().any():
        raise ValueError(f"Constructed survivor counts are negative for {sex}.")

    observed_q = d / initial_exposure
    if ((observed_q < 0) | (observed_q > 1)).any().any():
        raise ValueError(f"Observed death probabilities are outside [0, 1] for {sex}.")

    return initial_exposure, survivors, d


def save_matrix(matrix, filename):
    matrix.to_csv(DATA_DIR / filename, index=True, index_label="Year")


def two_factor_basis_functions(ages):
    ages = np.asarray(ages, dtype=float)
    phi1 = 1.0 - (ages - 18.0) / 82.0
    phi2 = (ages - 18.0) / 82.0
    return np.vstack([phi1, phi2])


def three_factor_basis_functions(ages):
    ages = np.asarray(ages, dtype=float)
    phi1 = np.where(ages <= 50, 1.0 - (ages - 18.0) / 32.0, 0.0)
    phi2 = np.where(ages <= 50, (ages - 18.0) / 32.0, 2.0 - ages / 50.0)
    phi3 = np.where(ages <= 50, 0.0, ages / 50.0 - 1.0)
    return np.vstack([phi1, phi2, phi3])


def negative_log_likelihood(factors, exposure, survivors, basis):
    """Binomial negative log-likelihood, excluding its constant term."""
    linear_predictor = np.asarray(factors, dtype=float) @ basis
    return float(
        np.sum(
            exposure * np.logaddexp(0.0, linear_predictor)
            - survivors * linear_predictor
        )
    )


def estimate_single_year(exposure, survivors, basis, initial_guess):
    objective = lambda factors: negative_log_likelihood(
        factors, exposure, survivors, basis
    )

    best_result = None
    for method in ("BFGS", "L-BFGS-B", "Nelder-Mead"):
        result = minimize(
            objective,
            x0=np.asarray(initial_guess, dtype=float),
            method=method,
            options={"maxiter": 5000},
        )
        if np.isfinite(result.fun) and (
            best_result is None or result.fun < best_result.fun
        ):
            best_result = result
        if result.success and np.isfinite(result.fun):
            break

    if best_result is None or not np.all(np.isfinite(best_result.x)):
        raise RuntimeError("Risk-factor optimization failed.")

    return best_result.x


def estimate_risk_factors(exposure, survivors, basis):
    """Estimate one risk-factor vector independently for every calendar year."""
    factor_names = [f"v{i}" for i in range(1, basis.shape[0] + 1)]
    estimates = np.empty((len(exposure.index), basis.shape[0]), dtype=float)
    initial_guess = np.zeros(basis.shape[0], dtype=float)

    for row_number, year in enumerate(exposure.index):
        estimates[row_number] = estimate_single_year(
            exposure.loc[year].to_numpy(dtype=float),
            survivors.loc[year].to_numpy(dtype=float),
            basis,
            initial_guess,
        )
        initial_guess = estimates[row_number]

    return pd.DataFrame(estimates, index=exposure.index, columns=factor_names)


def fitted_survival_matrix(factors, basis, ages):
    linear_predictor = factors.to_numpy(dtype=float) @ basis
    return pd.DataFrame(
        expit(linear_predictor),
        index=factors.index,
        columns=pd.Index(ages, name="Age"),
    )


def survival_long_table(probabilities, gender):
    result = (
        probabilities.rename_axis(index="Year", columns="Age")
        .stack()
        .rename("Survival Probability")
        .reset_index()
    )
    result["Gender"] = gender.lower()
    return result


def save_factor_plot(factors, sex_label, filename):
    factor_names = list(factors.columns)
    fig, axes = plt.subplots(1, len(factor_names), figsize=(9, 3.5), squeeze=False)

    for axis, factor_name in zip(axes[0], factor_names):
        axis.plot(factors.index, factors[factor_name], color="black", linewidth=0.8)
        axis.set_title(
            rf"${factor_name[0]}_{factor_name[1]}(t)$ vs. $t$" + f"\n({sex_label})",
            fontsize=9,
        )
        axis.set_xlabel("Year", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(DATA_DIR / filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


GRAY_CMAP = LinearSegmentedColormap.from_list(
    "ap_gray", [(0.8, 0.8, 0.8, 1.0), (0.3, 0.3, 0.3, 1.0)]
)


def save_surface_comparison(observed_p, fitted_p, sex_label, filename):
    ages = observed_p.columns.to_numpy(dtype=float)
    years = observed_p.index.to_numpy(dtype=float)
    age_grid, year_grid = np.meshgrid(ages, years)

    fig = plt.figure(figsize=(9, 3.5))
    panels = (
        (observed_p.to_numpy(dtype=float),
         rf"Observed $p^{{obs}}_{{x,t}}$, {sex_label.lower()}"),
        (fitted_p.to_numpy(dtype=float),
         rf"Fitted $\widehat{{p}}_{{x,t}}$, {sex_label.lower()}"),
    )

    for panel_number, (values, title) in enumerate(panels, start=1):
        axis = fig.add_subplot(1, 2, panel_number, projection="3d")
        axis.plot_surface(
            age_grid,
            year_grid,
            values,
            cmap=GRAY_CMAP,
            linewidth=0,
            antialiased=True,
            alpha=0.85,
        )
        axis.set_xlabel("Age")
        axis.set_ylabel("Year")
        axis.set_zlabel("Probability")
        axis.set_title(title)
        axis.set_zlim(0, 1)
        axis.set_xlim(ages.max(), ages.min())
        axis.set_ylim(years.min(), years.max())
        axis.set_xticks([18, 40, 60, 80, 100])
        axis.set_yticks([1956, 1970, 1990, 2020])
        axis.view_init(elev=20, azim=-125)

    fig.tight_layout()
    fig.savefig(DATA_DIR / filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fit_one_specification(exposure, survivors, ages, n_factors):
    basis = (
        two_factor_basis_functions(ages)
        if n_factors == 2
        else three_factor_basis_functions(ages)
    )
    factors = estimate_risk_factors(exposure, survivors, basis)
    fitted_p = fitted_survival_matrix(factors, basis, ages)
    return factors, fitted_p


def save_probability_output(probabilities, sex, n_factors, age_group):
    if age_group == "18":
        table = survival_long_table(probabilities, sex)
        table.to_csv(
            DATA_DIR / f"{sex.lower()}_prob_18_{n_factors}.csv",
            index=False,
            float_format="%.10f",
        )
    else:
        probabilities.to_csv(
            DATA_DIR / f"{sex.lower()}_prob_65_{n_factors}.csv",
            index=True,
            index_label="Year",
            float_format="%.10f",
        )


def main():
    exposure_18, deaths_18 = load_age_range(18, 100)
    exposure_65, deaths_65 = load_age_range(65, 90)

    matrices = {}
    for sex in ("Male", "Female"):
        e18, s18, d18 = prepare_working_matrices(
            exposure_18, deaths_18, sex, FULL_YEARS, AGES_18
        )
        e65, s65, d65 = prepare_working_matrices(
            exposure_65, deaths_65, sex, FULL_YEARS, AGES_65
        )

        matrices[(sex, "18")] = (e18, s18, d18)
        matrices[(sex, "65")] = (e65, s65, d65)

        for matrix, label in ((e18, "E"), (s18, "S"), (d18, "D")):
            save_matrix(matrix, f"{label}_{sex.lower()}_18.csv")
        for matrix, label in ((e65, "E"), (s65, "S"), (d65, "D")):
            save_matrix(matrix, f"{label}_{sex.lower()}_65.csv")

    for n_factors in (2, 3):
        for sex, sex_code in (("Male", "m"), ("Female", "f")):
            exposure, survivors, _ = matrices[(sex, "18")]
            factors, fitted_p = fit_one_specification(
                exposure, survivors, AGES_18, n_factors
            )
            factors.to_csv(
                DATA_DIR / f"v_18_{n_factors}{sex_code}.csv",
                index=True,
                index_label="Year",
            )
            save_probability_output(fitted_p, sex, n_factors, "18")
            save_factor_plot(
                factors,
                sex,
                f"v_{sex.lower()}_18_{n_factors}.pdf",
            )

            observed_p = survivors / exposure
            save_surface_comparison(
                observed_p,
                fitted_p,
                sex,
                f"comparison_{sex.lower()}_{n_factors}.pdf",
            )

    for n_factors in (2, 3):
        for sex, sex_code in (("Male", "m"), ("Female", "f")):
            exposure, survivors, _ = matrices[(sex, "65")]
            factors, fitted_p = fit_one_specification(
                exposure, survivors, AGES_65, n_factors
            )
            factors.to_csv(
                DATA_DIR / f"v_65_{n_factors}{sex_code}.csv",
                index=True,
                index_label="Year",
            )
            save_probability_output(fitted_p, sex, n_factors, "65")

    if RUN_ROBUSTNESS:
        for n_factors in (2, 3):
            for sex, sex_code in (("Male", "m"), ("Female", "f")):
                exposure, survivors, _ = matrices[(sex, "18")]
                short_exposure = exposure.loc[SHORT_YEARS]
                short_survivors = survivors.loc[SHORT_YEARS]
                short_factors, _ = fit_one_specification(
                    short_exposure, short_survivors, AGES_18, n_factors
                )
                short_factors.to_csv(
                    DATA_DIR / f"v_18_{n_factors}{sex_code}_short.csv",
                    index=True,
                    index_label="Year",
                )

    print("AP2/AP3 estimation completed; CSV and PDF outputs were saved to", DATA_DIR)


if __name__ == "__main__":
    main()