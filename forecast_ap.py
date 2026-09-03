#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


DATA_DIR = Path("/Users/sibo/Desktop/Data")
AGES = np.arange(18, 101)
TRAIN_YEARS = np.arange(1956, 2011)
VALIDATION_YEARS = np.arange(2011, 2021)

FACTOR_FILES = {
    "AP2": {
        "Male": DATA_DIR / "v_18_2m.csv",
        "Female": DATA_DIR / "v_18_2f.csv",
    },
    "AP3": {
        "Male": DATA_DIR / "v_18_3m.csv",
        "Female": DATA_DIR / "v_18_3f.csv",
    },
}


def load_factors(path, number_of_factors):
    factor_names = [f"v{i}" for i in range(1, number_of_factors + 1)]
    data = pd.read_csv(path)
    required = ["Year", *factor_names]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    data = data.loc[:, required].copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)
    data[factor_names] = data[factor_names].apply(pd.to_numeric, errors="raise")

    if data["Year"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate years.")

    data = data.set_index("Year").reindex(TRAIN_YEARS)
    if data.isna().any().any():
        raise ValueError(f"{path.name} must contain complete data for 1956--2010.")

    return data


def mortality_matrix(data, sex):
    required = {"Year", "Age", sex}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Mortality data are missing columns: {sorted(missing)}")

    selected = data.loc[:, ["Year", "Age", sex]].copy()
    selected["Year"] = pd.to_numeric(selected["Year"], errors="raise").astype(int)
    selected["Age"] = pd.to_numeric(selected["Age"], errors="raise").astype(int)
    selected[sex] = pd.to_numeric(selected[sex], errors="raise")

    if selected.duplicated(["Year", "Age"]).any():
        raise ValueError("Mortality data contain duplicate Year-Age rows.")

    years = np.concatenate((TRAIN_YEARS, VALIDATION_YEARS))
    matrix = selected.pivot(index="Year", columns="Age", values=sex)
    matrix = matrix.reindex(index=years, columns=AGES).astype(float)

    if matrix.isna().any().any():
        raise ValueError(
            f"Mortality data for {sex} must cover ages 18--100 and years 1956--2020."
        )

    return matrix


def load_observed_q(sex):
    deaths = mortality_matrix(
        pd.read_csv(DATA_DIR / "deat_total_18.csv"),
        sex,
    )
    central_exposure = mortality_matrix(
        pd.read_csv(DATA_DIR / "expo_total_18.csv"),
        sex,
    )

    if (deaths < 0).any().any():
        raise ValueError(f"Death counts must be non-negative for {sex}.")
    if (central_exposure <= 0).any().any():
        raise ValueError(f"Central exposures must be positive for {sex}.")

    initial_exposure = central_exposure + 0.5 * deaths
    observed_q = deaths / initial_exposure

    if ((observed_q < 0) | (observed_q > 1)).any().any():
        raise ValueError(f"Observed death probabilities are outside [0, 1] for {sex}.")

    return observed_q


def basis_matrix(model):
    ages = AGES.astype(float)

    if model == "AP2":
        phi1 = 1.0 - (ages - 18.0) / 82.0
        phi2 = (ages - 18.0) / 82.0
        return np.vstack((phi1, phi2))

    if model == "AP3":
        phi1 = np.where(ages <= 50.0, 1.0 - (ages - 18.0) / 32.0, 0.0)
        phi2 = np.where(ages <= 50.0, (ages - 18.0) / 32.0, 2.0 - ages / 50.0)
        phi3 = np.where(ages <= 50.0, 0.0, ages / 50.0 - 1.0)
        return np.vstack((phi1, phi2, phi3))

    raise ValueError(f"Unknown AP model: {model}")


def calibrate_random_walk(factors):
    increments = factors.diff().dropna()
    drift = increments.mean(axis=0)
    covariance = increments.cov()

    if not np.isfinite(drift.to_numpy(dtype=float)).all():
        raise ValueError("Risk-factor drift estimates must be finite.")

    standard_deviation = pd.Series(
        np.sqrt(np.diag(covariance.to_numpy(dtype=float))),
        index=factors.columns,
        name="StdDev",
    )
    if (
        not np.isfinite(standard_deviation.to_numpy(dtype=float)).all()
        or (standard_deviation <= 0).any()
    ):
        raise ValueError("Risk-factor increment standard deviations must be positive.")

    correlation = covariance.div(standard_deviation, axis=0).div(
        standard_deviation,
        axis=1,
    )
    np.fill_diagonal(correlation.values, 1.0)

    if not np.isfinite(correlation.to_numpy(dtype=float)).all():
        raise ValueError("Risk-factor increment correlations must be finite.")

    return drift, standard_deviation, correlation


def central_forecast(factors, drift, model):
    horizon = np.arange(1, len(VALIDATION_YEARS) + 1, dtype=float)[:, None]
    forecast_factors = factors.iloc[-1].to_numpy(dtype=float) + (
        horizon * drift.to_numpy(dtype=float)
    )
    logit_p = forecast_factors @ basis_matrix(model)
    fitted_p = expit(logit_p)
    fitted_q = 1.0 - fitted_p

    return pd.DataFrame(
        fitted_q,
        index=VALIDATION_YEARS,
        columns=AGES,
    )


def compute_naive_mae(observed_q):
    training_q = observed_q.loc[TRAIN_YEARS, AGES].to_numpy(dtype=float)
    if not np.isfinite(training_q).all():
        raise ValueError("Training death probabilities must be finite.")

    naive_mae = float(np.mean(np.abs(np.diff(training_q, axis=0))))
    if not np.isfinite(naive_mae) or naive_mae <= 0:
        raise ValueError("The observed-data naive benchmark must be positive.")
    return naive_mae


def compute_accuracy(observed_q, forecast_q, naive_mae):
    validation_q = observed_q.loc[VALIDATION_YEARS, AGES].to_numpy(dtype=float)
    forecast_values = forecast_q.loc[VALIDATION_YEARS, AGES].to_numpy(dtype=float)

    if validation_q.shape != forecast_values.shape:
        raise ValueError("Observed and forecast probability matrices are not aligned.")
    if not np.isfinite(validation_q).all() or not np.isfinite(forecast_values).all():
        raise ValueError("Observed and forecast death probabilities must be finite.")

    mae_raw = float(np.mean(np.abs(forecast_values - validation_q)))
    return 100.0 * mae_raw, mae_raw / naive_mae


def correlation_to_long(correlation, model, sex):
    return (
        correlation.rename_axis(index="Factor_1", columns="Factor_2")
        .stack()
        .rename("Correlation")
        .reset_index()
        .assign(Model=model, Sex=sex)[
            ["Model", "Sex", "Factor_1", "Factor_2", "Correlation"]
        ]
    )


def main():
    drift_tables = []
    correlation_tables = []
    error_rows = []

    observed_q = {
        sex: load_observed_q(sex)
        for sex in ("Male", "Female")
    }
    naive_mae = {
        sex: compute_naive_mae(observed_q[sex])
        for sex in ("Male", "Female")
    }

    for model, files_by_sex in FACTOR_FILES.items():
        number_of_factors = int(model[-1])

        for sex, factor_file in files_by_sex.items():
            factors = load_factors(factor_file, number_of_factors)
            drift, standard_deviation, correlation = calibrate_random_walk(factors)
            forecast_q = central_forecast(factors, drift, model)
            mae, mase = compute_accuracy(
                observed_q[sex],
                forecast_q,
                naive_mae[sex],
            )

            drift_tables.append(
                pd.concat(
                    (drift.rename("Drift"), standard_deviation),
                    axis=1,
                )
                .rename_axis("Factor")
                .reset_index()
                .assign(Model=model, Sex=sex)[
                    ["Model", "Sex", "Factor", "Drift", "StdDev"]
                ]
            )
            correlation_tables.append(
                correlation_to_long(correlation, model, sex)
            )
            error_rows.append(
                {
                    "Model": model,
                    "Sex": sex,
                    "MAE": mae,
                    "MASE": mase,
                }
            )

    drift_parameters = pd.concat(drift_tables, ignore_index=True).sort_values(
        ["Sex", "Model", "Factor"],
        ignore_index=True,
    )
    correlations = pd.concat(correlation_tables, ignore_index=True).sort_values(
        ["Sex", "Model", "Factor_1", "Factor_2"],
        ignore_index=True,
    )
    errors = pd.DataFrame(error_rows).sort_values(
        ["Sex", "Model"],
        ignore_index=True,
    )

    drift_parameters.to_csv(
        DATA_DIR / "ap_rw_drift.csv",
        index=False,
        float_format="%.4f",
    )
    correlations.to_csv(
        DATA_DIR / "ap_rw_correlation.csv",
        index=False,
        float_format="%.4f",
    )
    errors.to_csv(
        DATA_DIR / "ap_oos_errors.csv",
        index=False,
        float_format="%.4f",
    )


if __name__ == "__main__":
    main()

# =============================================================================
# Chapter 6.2 -- Conditional long-term projections, 2021--2040
# =============================================================================

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.linalg import cholesky


FULL_YEARS = np.arange(1956, 2021)
PROJECTION_YEARS = np.arange(2021, 2041)
REFERENCE_AGES = (65, 75, 85)
N_SIM = 1000
PROJECTION_SEED = 20260825


def load_factors_full(path, number_of_factors):
    factor_names = [f"v{i}" for i in range(1, number_of_factors + 1)]
    data = pd.read_csv(path)
    required = ["Year", *factor_names]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    data = data.loc[:, required].copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)
    data[factor_names] = data[factor_names].apply(
        pd.to_numeric,
        errors="raise",
    )
    if data["Year"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate years.")

    data = data.set_index("Year").reindex(FULL_YEARS)
    if data.isna().any().any():
        raise ValueError(f"{path.name} must contain complete data for 1956--2020.")
    return data


def calibrate_random_walk_full(factors):
    increments = factors.diff().dropna()
    drift = increments.mean(axis=0)
    covariance = increments.cov()
    covariance_values = covariance.to_numpy(dtype=float)
    covariance_values = 0.5 * (covariance_values + covariance_values.T)

    if not np.isfinite(drift.to_numpy(dtype=float)).all():
        raise ValueError("Full-sample drift estimates must be finite.")
    if not np.isfinite(covariance_values).all():
        raise ValueError("Full-sample covariance estimates must be finite.")

    standard_deviation = pd.Series(
        np.sqrt(np.diag(covariance_values)),
        index=factors.columns,
        name="StdDev",
    )
    if (standard_deviation <= 0).any():
        raise ValueError("Full-sample standard deviations must be positive.")

    correlation_values = covariance_values / np.outer(
        standard_deviation.to_numpy(dtype=float),
        standard_deviation.to_numpy(dtype=float),
    )
    np.fill_diagonal(correlation_values, 1.0)
    correlation = pd.DataFrame(
        correlation_values,
        index=factors.columns,
        columns=factors.columns,
    )
    covariance = pd.DataFrame(
        covariance_values,
        index=factors.columns,
        columns=factors.columns,
    )
    return drift, standard_deviation, covariance, correlation


def stable_cholesky(covariance):
    covariance_values = covariance.to_numpy(dtype=float)
    covariance_values = 0.5 * (covariance_values + covariance_values.T)

    try:
        return cholesky(covariance_values, lower=True), 0.0
    except np.linalg.LinAlgError:
        scale = max(float(np.mean(np.diag(covariance_values))), 1.0)
        jitter = scale * 1e-12
        identity = np.eye(covariance_values.shape[0])

        for _ in range(12):
            try:
                adjusted = covariance_values + jitter * identity
                return cholesky(adjusted, lower=True), jitter
            except np.linalg.LinAlgError:
                jitter *= 10.0

    raise ValueError("The covariance matrix could not be Cholesky-factorized.")


def simulate_ap_projection(factors, drift, c_factor, model, rng):
    horizon = len(PROJECTION_YEARS)
    number_of_factors = factors.shape[1]
    basis = basis_matrix(model)
    last_factor = factors.iloc[-1].to_numpy(dtype=float)
    drift_values = drift.to_numpy(dtype=float)

    steps = np.arange(1, horizon + 1, dtype=float)[:, None]
    central_factors = last_factor + steps * drift_values
    central_logit_p = central_factors @ basis
    central_q = 1.0 - expit(central_logit_p)

    gaussian_draws = rng.standard_normal(
        size=(N_SIM, horizon, number_of_factors)
    )
    innovations = gaussian_draws @ c_factor.T
    simulated_increments = innovations + drift_values[None, None, :]
    simulated_factors = last_factor[None, None, :] + np.cumsum(
        simulated_increments,
        axis=1,
    )
    simulated_logit_p = np.einsum(
        "shk,ka->sha",
        simulated_factors,
        basis,
    )
    simulated_q = 1.0 - expit(simulated_logit_p)

    quantiles = np.quantile(
        simulated_q,
        q=(0.025, 0.10, 0.25, 0.75, 0.90, 0.975),
        axis=0,
        method="median_unbiased",
    )
    return central_q, quantiles


def projection_to_long(central_q, quantiles, model, sex):
    return pd.DataFrame(
        {
            "Model": model,
            "Sex": sex,
            "Age": np.repeat(AGES, len(PROJECTION_YEARS)),
            "Year": np.tile(PROJECTION_YEARS, len(AGES)),
            "Central": central_q.T.reshape(-1),
            "Lower_50": quantiles[2].T.reshape(-1),
            "Upper_50": quantiles[3].T.reshape(-1),
            "Lower_80": quantiles[1].T.reshape(-1),
            "Upper_80": quantiles[4].T.reshape(-1),
            "Lower_95": quantiles[0].T.reshape(-1),
            "Upper_95": quantiles[5].T.reshape(-1),
        }
    )


def matrix_to_long(matrix, value_name, model, sex):
    rows = []
    for factor_1 in matrix.index:
        for factor_2 in matrix.columns:
            rows.append(
                {
                    "Model": model,
                    "Sex": sex,
                    "Factor_1": factor_1,
                    "Factor_2": factor_2,
                    value_name: float(matrix.loc[factor_1, factor_2]),
                }
            )
    return pd.DataFrame(rows)


def plot_ap_projection(model, projection_summary, observed_q):
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5), sharex=True)

    for row, sex in enumerate(("Male", "Female")):
        for column, age in enumerate(REFERENCE_AGES):
            ax = axes[row, column]
            panel = projection_summary.loc[
                (projection_summary["Model"] == model)
                & (projection_summary["Sex"] == sex)
                & (projection_summary["Age"] == age)
            ].sort_values("Year")

            historical = observed_q[sex].loc[FULL_YEARS, age]
            ax.plot(
                FULL_YEARS,
                historical,
                color="0.55",
                linewidth=0.8,
                label="Observed",
            )
            ax.fill_between(
                panel["Year"],
                panel["Lower_95"],
                panel["Upper_95"],
                color="#4C78A8",
                alpha=0.16,
                linewidth=0,
                label="95%",
            )
            ax.fill_between(
                panel["Year"],
                panel["Lower_80"],
                panel["Upper_80"],
                color="#4C78A8",
                alpha=0.28,
                linewidth=0,
                label="80%",
            )
            ax.fill_between(
                panel["Year"],
                panel["Lower_50"],
                panel["Upper_50"],
                color="#4C78A8",
                alpha=0.43,
                linewidth=0,
                label="50%",
            )
            ax.plot(
                panel["Year"],
                panel["Central"],
                color="#1F4E79",
                linewidth=1.2,
                label="Central",
            )
            ax.axvline(2020.5, color="firebrick", linestyle=":", linewidth=0.8)
            ax.set_title(f"{sex}, age {age}", fontsize=9)
            ax.set_xlabel("Year", fontsize=8)
            ax.set_ylabel(r"$q_{x,t}$", fontsize=8)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [0, 4, 3, 2, 1]
    fig.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(DATA_DIR / f"projection_{model}.pdf", format="pdf")
    plt.close(fig)


def main_long_term_projection():
    observed_q = {
        sex: load_observed_q(sex)
        for sex in ("Male", "Female")
    }
    rng = np.random.default_rng(PROJECTION_SEED)

    projection_tables = []
    drift_tables = []
    correlation_tables = []
    cholesky_tables = []
    diagnostic_rows = []

    for model, files_by_sex in FACTOR_FILES.items():
        number_of_factors = int(model[-1])

        for sex, factor_file in files_by_sex.items():
            factors = load_factors_full(factor_file, number_of_factors)
            drift, standard_deviation, covariance, correlation = (
                calibrate_random_walk_full(factors)
            )
            c_factor, jitter = stable_cholesky(covariance)
            central_q, quantiles = simulate_ap_projection(
                factors,
                drift,
                c_factor,
                model,
                rng,
            )
            projection_tables.append(
                projection_to_long(central_q, quantiles, model, sex)
            )

            drift_tables.append(
                pd.concat(
                    (drift.rename("Drift"), standard_deviation),
                    axis=1,
                )
                .rename_axis("Factor")
                .reset_index()
                .assign(Model=model, Sex=sex)[
                    ["Model", "Sex", "Factor", "Drift", "StdDev"]
                ]
            )
            correlation_tables.append(
                matrix_to_long(
                    correlation,
                    "Correlation",
                    model,
                    sex,
                )
            )
            cholesky_tables.append(
                matrix_to_long(
                    pd.DataFrame(
                        c_factor,
                        index=factors.columns,
                        columns=factors.columns,
                    ),
                    "Cholesky",
                    model,
                    sex,
                )
            )
            diagnostic_rows.append(
                {
                    "Model": model,
                    "Sex": sex,
                    "Diagonal_Adjustment": jitter,
                }
            )

    projection_summary = pd.concat(
        projection_tables,
        ignore_index=True,
    ).sort_values(["Sex", "Model", "Age", "Year"], ignore_index=True)
    drift_full = pd.concat(
        drift_tables,
        ignore_index=True,
    ).sort_values(["Sex", "Model", "Factor"], ignore_index=True)
    correlation_full = pd.concat(
        correlation_tables,
        ignore_index=True,
    ).sort_values(
        ["Sex", "Model", "Factor_1", "Factor_2"],
        ignore_index=True,
    )
    cholesky_full = pd.concat(
        cholesky_tables,
        ignore_index=True,
    ).sort_values(
        ["Sex", "Model", "Factor_1", "Factor_2"],
        ignore_index=True,
    )
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["Sex", "Model"],
        ignore_index=True,
    )

    projection_summary.to_csv(
        DATA_DIR / "ap_projection_summary.csv",
        index=False,
        float_format="%.4f",
    )
    projection_summary.loc[
        (projection_summary["Year"] == 2040)
        & (projection_summary["Age"].isin(REFERENCE_AGES))
    ].to_csv(
        DATA_DIR / "projection_2040_ap.csv",
        index=False,
        float_format="%.4f",
    )
    drift_full.to_csv(
        DATA_DIR / "ap_rw_drift_full.csv",
        index=False,
        float_format="%.4f",
    )
    correlation_full.to_csv(
        DATA_DIR / "ap_rw_correlation_full.csv",
        index=False,
        float_format="%.4f",
    )
    cholesky_full.to_csv(
        DATA_DIR / "ap_rw_cholesky_full.csv",
        index=False,
        float_format="%.4f",
    )
    diagnostics.to_csv(
        DATA_DIR / "ap_cholesky_diagnostics.csv",
        index=False,
        float_format="%.4f",
    )

    for model in FACTOR_FILES:
        plot_ap_projection(model, projection_summary, observed_q)


if __name__ == "__main__":
    main_long_term_projection()
