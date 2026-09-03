#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 15:38:36 2025

@author: sibo
Robustness plots: compare v(t) estimated on 1956–2020 (long) vs independently
re-estimated on 1956–2000 (short). For year-by-year estimation, these coincide
on overlapping years by construction.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family':          'serif',
    'font.size':             8,
    'axes.titlesize':       10,
    'axes.labelsize':        6,
    'xtick.labelsize':       5,
    'ytick.labelsize':       5,
    'axes.facecolor':       'white',
    'figure.facecolor':     'white',
    'axes.edgecolor':       'black',
    'axes.linewidth':        0.8,
    'xtick.major.size':      2,
    'ytick.major.size':      2,
    'xtick.major.width':     0.6,
    'ytick.major.width':     0.6,
})

LCOLOR = 'blue'
SCOLOR = 'red'
LW = 0.5
LABEL_L = '1956\u20132020'
LABEL_S = '1956\u20132000'


def style_ax(ax, title, show_xlabel=True):
    ax.set_title(title, ha='center', size=10, pad=4)
    if show_xlabel:
        ax.set_xlabel('Year', size=6, labelpad=3)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel('')
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=5, pad=2, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('black')


# ── Two-factor ───────────────────────────────────────────────────────────────
v_18_2m_long = pd.read_csv('/Users/sibo/Desktop/Data/v_18_2m.csv').set_index('Year')
v_18_2f_long = pd.read_csv('/Users/sibo/Desktop/Data/v_18_2f.csv').set_index('Year')
v_18_2m_short = pd.read_csv('/Users/sibo/Desktop/Data/v_18_2m_short.csv').set_index('Year')
v_18_2f_short = pd.read_csv('/Users/sibo/Desktop/Data/v_18_2f_short.csv').set_index('Year')

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
axes[0, 2].axis('off')
axes[1, 2].axis('off')

for col, v_col, title in [
    (0, 'v1', r'$v_1(t)$ (Male)'),
    (1, 'v2', r'$v_2(t)$ (Male)'),
]:
    ax = axes[0, col]
    ax.plot(v_18_2m_long.index, v_18_2m_long[v_col], color=LCOLOR, lw=LW, label=LABEL_L)
    ax.plot(v_18_2m_short.index, v_18_2m_short[v_col], color=SCOLOR, lw=LW, linestyle='--', label=LABEL_S)
    style_ax(ax, title, show_xlabel=False)

for col, v_col, title in [
    (0, 'v1', r'$v_1(t)$ (Female)'),
    (1, 'v2', r'$v_2(t)$ (Female)'),
]:
    ax = axes[1, col]
    ax.plot(v_18_2f_long.index, v_18_2f_long[v_col], color=LCOLOR, lw=LW, label=LABEL_L)
    ax.plot(v_18_2f_short.index, v_18_2f_short[v_col], color=SCOLOR, lw=LW, linestyle='--', label=LABEL_S)
    style_ax(ax, title, show_xlabel=True)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2,
           fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))

plt.subplots_adjust(hspace=0.38, wspace=0.22,
                    left=0.06, right=0.97,
                    top=0.93, bottom=0.10)
plt.savefig('/Users/sibo/Desktop/Data/robustness_2.pdf', dpi=150, bbox_inches='tight')
plt.show()

# ── Three-factor ───────────────────────────────────────────────────────────────
v_18_3m_long = pd.read_csv('/Users/sibo/Desktop/Data/v_18_3m.csv').set_index('Year')
v_18_3f_long = pd.read_csv('/Users/sibo/Desktop/Data/v_18_3f.csv').set_index('Year')
v_18_3m_short = pd.read_csv('/Users/sibo/Desktop/Data/v_18_3m_short.csv').set_index('Year')
v_18_3f_short = pd.read_csv('/Users/sibo/Desktop/Data/v_18_3f_short.csv').set_index('Year')

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))

for col, v_col, title in [
    (0, 'v1', r'$v_1(t)$ (Male)'),
    (1, 'v2', r'$v_2(t)$ (Male)'),
    (2, 'v3', r'$v_3(t)$ (Male)'),
]:
    ax = axes[0, col]
    ax.plot(v_18_3m_long.index, v_18_3m_long[v_col], color=LCOLOR, lw=LW, label=LABEL_L)
    ax.plot(v_18_3m_short.index, v_18_3m_short[v_col], color=SCOLOR, lw=LW, linestyle='--', label=LABEL_S)
    style_ax(ax, title, show_xlabel=False)

for col, v_col, title in [
    (0, 'v1', r'$v_1(t)$ (Female)'),
    (1, 'v2', r'$v_2(t)$ (Female)'),
    (2, 'v3', r'$v_3(t)$ (Female)'),
]:
    ax = axes[1, col]
    ax.plot(v_18_3f_long.index, v_18_3f_long[v_col], color=LCOLOR, lw=LW, label=LABEL_L)
    ax.plot(v_18_3f_short.index, v_18_3f_short[v_col], color=SCOLOR, lw=LW, linestyle='--', label=LABEL_S)
    style_ax(ax, title, show_xlabel=True)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2,
           fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))

plt.subplots_adjust(hspace=0.38, wspace=0.22,
                    left=0.06, right=0.97,
                    top=0.93, bottom=0.10)
plt.savefig('/Users/sibo/Desktop/Data/robustness_3.pdf', dpi=150, bbox_inches='tight')
plt.show()
