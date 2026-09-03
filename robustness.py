#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:26:15 2026

@author: sibo
"""
"""Create AP2 and AP3 robustness plots for two estimation windows."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("/Users/sibo/Desktop/Data")
LONG_YEARS = range(1956, 2021)
SHORT_YEARS = range(1956, 2001)

LONG_COLOR = "blue"
SHORT_COLOR = "red"
LINE_WIDTH = 0.7
LONG_LABEL = "1956-2020"
SHORT_LABEL = "1956-2000"
FIGURE_WIDTH = 9.0
FIGURE_HEIGHT = 5.5


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
    }
)


def read_factors(filename, expected_years, number_of_factors):
    data = pd.read_csv(DATA_DIR / filename)
    required = {"Year", *(f"v{i}" for i in range(1, number_of_factors + 1))}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

    data = data.loc[
        :,
        ["Year", *(f"v{i}" for i in range(1, number_of_factors + 1))],
    ].copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)

    if data["Year"].duplicated().any():
        raise ValueError(f"{filename} contains duplicate years.")

    factor_columns = [f"v{i}" for i in range(1, number_of_factors + 1)]
    data[factor_columns] = data[factor_columns].apply(pd.to_numeric, errors="raise")
    data = data.set_index("Year").reindex(expected_years)

    if data.isna().any().any():
        raise ValueError(f"{filename} has missing values after year alignment.")

    return data


def style_axis(axis, title, show_x_label):
    axis.set_title(title, pad=4)
    axis.set_xlabel("Year" if show_x_label else "")
    axis.set_ylabel("")
    axis.grid(False)
    axis.tick_params(top=False, right=False)
    if not show_x_label:
        axis.tick_params(axis="x", labelbottom=False)


def create_robustness_plot(number_of_factors):
    long_data = {
        "Male": read_factors(
            f"v_18_{number_of_factors}m.csv",
            LONG_YEARS,
            number_of_factors,
        ),
        "Female": read_factors(
            f"v_18_{number_of_factors}f.csv",
            LONG_YEARS,
            number_of_factors,
        ),
    }
    short_data = {
        "Male": read_factors(
            f"v_18_{number_of_factors}m_short.csv",
            SHORT_YEARS,
            number_of_factors,
        ),
        "Female": read_factors(
            f"v_18_{number_of_factors}f_short.csv",
            SHORT_YEARS,
            number_of_factors,
        ),
    }

    figure, axes = plt.subplots(
        2,
        number_of_factors,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        squeeze=False,
        sharex=True,
    )

    for row, sex in enumerate(("Male", "Female")):
        for column in range(number_of_factors):
            factor = f"v{column + 1}"
            axis = axes[row, column]
            axis.plot(
                long_data[sex].index,
                long_data[sex][factor],
                color=LONG_COLOR,
                linewidth=LINE_WIDTH,
                label=LONG_LABEL,
            )
            axis.plot(
                short_data[sex].index,
                short_data[sex][factor],
                color=SHORT_COLOR,
                linewidth=LINE_WIDTH,
                linestyle="--",
                label=SHORT_LABEL,
            )
            style_axis(
                axis,
                rf"$v_{{{column + 1}}}(t)$ ({sex})",
                show_x_label=(row == 1),
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.subplots_adjust(
        hspace=0.30,
        wspace=0.20,
        left=0.07,
        right=0.98,
        bottom=0.14,
        top=0.95,
    )
    figure.savefig(
        DATA_DIR / f"robustness_AP{number_of_factors}.pdf",
        format="pdf",
    )
    plt.close(figure)


def main():
    for number_of_factors in (2, 3):
        create_robustness_plot(number_of_factors)


if __name__ == "__main__":
    main()
