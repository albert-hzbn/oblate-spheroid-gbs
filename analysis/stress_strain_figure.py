#!/usr/bin/env python3
"""Stress-strain curves of the nanocrystalline Cu tension runs (SI figure)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[1]
R = REPO
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dd'
seq = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#2a78d6', '#256abf', '#184f95', '#104281', '#0d366b']
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
ds = [3, 4, 5, 6, 8, 10, 12, 14, 16]
for k, d in enumerate(ds):
    x = np.loadtxt(R / 'results' / 'polycrystal' / f'd{d}_T300_r5e8' / 'stress.dat', skiprows=1)
    a.plot(x[:, 1] * 100, x[:, 2], color=seq[min(k, len(seq) - 1)] if k else '#9ec5f4', lw=1.4, label=f'{d} nm')
a.axvspan(6, 10, color=GRID, alpha=0.6, lw=0)
a.text(8, 0.15, 'flow-stress\nwindow', ha='center', fontsize=7, color=MUTED)
a.set(xlabel='tensile strain (%)', ylabel='tensile stress (GPa)', xlim=(0, 10), ylim=(0, 3.7), yticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5])
a.legend(loc='upper left', ncol=3, columnspacing=0.8, handlelength=1.2, title='grain size, 300 K, $5\\times10^8$ s$^{-1}$', title_fontsize=7)
cols = {300: '#2a78d6', 500: '#eb6834', 700: '#1baf7a'}
lss = {'0.0003': ':', '0.001': '--', '0.003': '-'}
for T, c in cols.items():
    for r, ls in lss.items():
        f = R / 'results' / 'polycrystal' / f'rate_d5_T{T}_r{r}' / 'stress.dat'
        x = np.loadtxt(f, skiprows=1)
        b.plot(x[:, 1] * 100, x[:, 2], color=c, ls=ls, lw=1.3,
               label=f'{T} K, {float(r)*1e12:.0e} s$^{{-1}}$'.replace('e+0', 'e'))
b.set(xlabel='tensile strain (%)', ylabel='tensile stress (GPa)', xlim=(0, 10), ylim=(0, 3.7), yticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5])
b.legend(loc='upper left', ncol=3, columnspacing=0.6, handlelength=1.6, fontsize=6.2, title='$d$ = 5 nm', title_fontsize=7)
for ax, l in ((a, 'a'), (b, 'b')):
    ax.text(-0.15, 1.04, f'({l})', transform=ax.transAxes, fontweight='bold', va='bottom')
fig.tight_layout(); fig.savefig(R / 'figures' / 'si' / 'figS_stress_strain.pdf'); print('wrote figS_stress_strain.pdf')
