#!/usr/bin/env python3
"""Corrected replacements for the analysis figures withdrawn after the 2026-09-24 audit.

Every curve follows an equation of the revised manuscript (noted per figure);
inputs are the manuscript's illustrative values, not fitted or material-validated
parameters. Output: ../figures/fig_*.pdf
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'figures' / 'theory'
ROWS = HERE / 'data' / 'systems_41.json'
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical order (dataviz reference palette, light mode).
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dd'
LS = ['-', '--', '-.', ':', (0, (5, 1, 1, 1))]
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'legend.fontsize': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 2.0, 'legend.frameon': False, 'pdf.fonttype': 42,
})

kB = 8.617333262e-5          # eV/K
EV = 1.602176634e-19         # J
G_CU, B_CU, NU = 48.3e9, 0.2556e-9, 1 / 3


def panel(ax, letter):
    ax.text(-0.16, 1.04, f'({letter})', transform=ax.transAxes,
            fontweight='bold', va='bottom')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print('wrote', name)


# --------------------------------------------------------------------------
# Eshelby S1313 for an oblate spheroid a1=a2=1, a3=alpha (same quadrature as
# scripts/audit_revised_manuscript.py, which checks sphere and thin-disk limits).
def s1313(alpha, nu):
    ax2 = np.array([1., 1., alpha ** 2])

    def integral(i, j=None):
        f = lambda q: 1 / ((ax2[i] + q) * np.sqrt(np.prod(ax2 + q))
                           * (1 if j is None else ax2[j] + q))
        bounds = sorted({0., alpha ** 2, 1.}) + [np.inf]
        return 2 * np.pi * alpha * sum(quad(f, lo, hi, epsabs=1e-11)[0]
                                       for lo, hi in zip(bounds[:-1], bounds[1:]))
    i1, i3, i13 = integral(0), integral(2), integral(0, 2)
    return ((1 + alpha ** 2) * i13 + (1 - 2 * nu) * (i1 + i3)) / (16 * np.pi * (1 - nu))


def fig_s1313():
    alphas = np.linspace(0.02, 1.0, 60)
    nus = [0.25, 0.30, 1 / 3, 0.40]
    labels = ['ν = 0.25', 'ν = 0.30', 'ν = 1/3', 'ν = 0.40']
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for k, (nu, lab) in enumerate(zip(nus, labels)):
        s = np.array([s1313(al, nu) for al in alphas])
        a.plot(alphas, s, color=C[k], ls=LS[k], label=lab)
        b.plot(alphas, 1 - 2 * s, color=C[k], ls=LS[k], label=lab)
    s0 = s1313(0.5, NU)
    a.axhline(0.5, color=MUTED, lw=0.8, ls=':')
    a.text(0.98, 0.49, 'thin-disk limit 1/2', ha='right', va='top', fontsize=7.5, color=MUTED)
    a.set(xlabel='aspect ratio α = c/a', ylabel='$S_{1313}$', xlim=(0, 1.02))
    b.set(xlabel='aspect ratio α = c/a', ylabel='$\\beta_1 = 1-2S_{1313}$', xlim=(0, 1.02), ylim=(0, None))
    a.legend(loc='upper right', bbox_to_anchor=(1.0, 0.86))
    # Mark the value used in the paper: dotted drop lines to both axes and a
    # label beside the point (no leader line).
    for ax, y, txt, dy, va in ((a, s0, f'$S_{{1313}}$ = {s0:.4f}', 0.032, 'bottom'),
                               (b, 1 - 2 * s0, f'$\\beta_1$ = {1 - 2 * s0:.4f}', -0.07, 'top')):
        x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
        ax.plot([0.5, 0.5], [y0, y], color=MUTED, lw=0.9, ls=':', zorder=1)
        ax.plot([x0, 0.5], [y, y], color=MUTED, lw=0.9, ls=':', zorder=1)
        ax.plot([0.5], [y], 'o', ms=6, color=INK, mec='white', mew=1.5, zorder=5)
        ax.text(0.53, y + dy, txt + '\n(α = 1/2, ν = 1/3)', fontsize=8, va=va, ha='left',
                linespacing=1.3, zorder=6)
        ax.set_xlim(x0, ax.get_xlim()[1]); ax.set_ylim(y0, ax.get_ylim()[1])
    panel(a, 'a'); panel(b, 'b')
    save(fig, 'fig_S1313_aspect.pdf')
    return s0


def fig_decay():
    r = np.logspace(0, np.log10(20), 200)
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.axvspan(1, 2, color=GRID, alpha=0.7, lw=0)
    ax.text(1.06, 1.2e-2, 'near field:\npoint-dipole\nform not\nvalid', fontsize=7, color=MUTED, va='top')
    ax.loglog(r, 1 / r ** 3, color=C[0], ls=LS[0], label='inclusion dipole, $(W/r)^3$')
    ax.loglog(r, 1 / r, color=C[1], ls=LS[1], label='straight line defect, $W/r$')
    for f, k in ((1 / 125, 0), (1 / 5, 1)):
        ax.plot([5], [f], 'o', ms=6, color=C[k], mec='white', mew=1.5, zorder=5)
    ax.annotate('0.8% at r = 5W', (5, 1 / 125), xytext=(2.3, 2.5e-4), textcoords='data', fontsize=7.5,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8, shrinkB=4))
    ax.annotate('20% at r = 5W', (5, 0.2), xytext=(7.5, 0.025), textcoords='data', fontsize=7.5,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8, shrinkB=4))
    ax.set(xlabel='distance r / W', ylabel='stress scale, normalised at r = W', ylim=(1e-4, 3))
    ax.legend(loc='upper right', fontsize=7)
    save(fig, 'fig_dipole_vs_line_decay.pdf')


def fig_kinetics():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    x = np.logspace(-3, np.log10(5), 300)
    a.loglog(x, 2 * np.sinh(x), color=C[0], ls=LS[0], label='net, $2\\sinh x$')
    a.loglog(x, np.exp(x), color=C[1], ls=LS[1], label='forward only, $e^x$')
    a.loglog(x, 2 * x, color=C[2], ls=LS[2], label='linear limit, $2x$')
    x0 = 0.0207
    a.axvline(x0, color=MUTED, lw=0.8, ls=':')
    a.text(x0 * 1.6, 2.5e-3, 'worked example\nx = 0.0207', fontsize=7.5, color=MUTED)
    a.set(xlabel='$x=\\Delta w/kT=\\tau_d\\gamma_0V_0/(2kT)$', ylabel='rate factor')
    a.legend(loc='upper left', fontsize=7)
    # Eq. (rate): gamma_dot = (2 eta W g0 nu0/d) exp(-F/kT) sinh(tau_d g0 V0 / 2kT)
    W, g0, d, T, F = 0.75e-9, 0.10, 10e-6, 773.0, 0.40512
    V0 = 2 * np.pi * W ** 3 * 0.5 / 3
    tau = np.logspace(5, 8, 200)
    coef = 2 * W * g0 / d * np.exp(-F / (kB * T)) * np.sinh(tau * g0 * V0 / (2 * kB * T * EV))
    b.axhspan(1e-3, 1e-1, color=GRID, alpha=0.8, lw=0)
    nu0 = kB * T / 4.135667696e-15          # transition-state frequency kT/h, s^-1
    for k, eta in enumerate([1e-7, 1e-6, 1e-5]):
        b.loglog(tau / 1e6, eta * nu0 * coef, color=C[k], ls=LS[k],
                 label=f'$\\eta$ = 10$^{{{int(np.log10(eta))}}}$')
    req = 1e-1 / (7.0916e-10 * nu0)
    b.text(0.97, 0.04, f'$\\nu_0=kT/h$ = {nu0:.2e} s$^{{-1}}$ at 773 K;\n0.1 s$^{{-1}}$ at 5 MPa needs $\\eta$ = {req:.1e}',
           transform=b.transAxes, ha='right', fontsize=7)
    b.set(xlabel='driving stress $\\tau_d$ (MPa)', ylabel='$\\dot\\gamma$ (s$^{-1}$)', ylim=(1e-7, 1e4), xlim=(0.1, 100))
    b.legend(loc='upper left', fontsize=7)
    panel(a, 'a'); panel(b, 'b')
    save(fig, 'fig_kinetics_sinh.pdf')


def fig_meanfield():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    xs = np.linspace(-8, 8, 400)          # x = (tau g0 V0 - E)/kT
    for k, j in enumerate([0., 2., 3.5]):  # j = J_Sigma / kT
        n = []
        g = 0.0
        for xv in xs:                        # continuation from low field
            for _ in range(400):
                g = 1 / (1 + np.exp(-(xv + j * g)))
            n.append(g)
        a.plot(xs, n, color=C[k], ls=LS[k], label=f'$J_\\Sigma/kT$ = {j:g}')
    a.annotate('coupling lowers\nthe stress (softening)', (-2.2, 0.6),
               xytext=(-8.3, 0.80), fontsize=7.5, arrowprops=dict(arrowstyle='->', color=MUTED))
    a.set(xlabel='reduced field $(\\tau\\gamma_0V_0-E)/kT$', ylabel='mean occupation $\\bar n$')
    a.legend(loc='lower right')
    rho = np.logspace(12, 16, 100)
    z, g0, W = 6, 0.10, B_CU
    soft = z * g0 * G_CU * np.pi * W ** 2 / (6 * (1 - NU)) * rho
    taylor = 0.3 * G_CU * B_CU * np.sqrt(rho)
    b.loglog(rho, soft / 1e6, color=C[0], ls=LS[0], label='|Δτ$_{MF}$|, mean field: softening for\n$J_\\Sigma>0$, hardening for $J_\\Sigma<0$')
    b.loglog(rho, taylor / 1e6, color=C[1], ls=LS[1], label='classical Taylor, $0.3\\,Gb\\sqrt{\\rho}$\n(sign: hardening)')
    b.set(xlabel='dislocation density ρ (m$^{-2}$)', ylabel='stress magnitude (MPa)', ylim=(1e-3, 1e6))
    b.legend(loc='upper left', fontsize=7)
    panel(a, 'a'); panel(b, 'b')
    save(fig, 'fig_meanfield_softening.pdf')


def fig_estimates():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    a, b, c = axs
    q = np.linspace(0.05, 2, 200)
    a.semilogy(q, np.exp(-2 * np.pi * q), color=C[0])
    for qq in (0.5, 1.0):
        v = np.exp(-2 * np.pi * qq)
        a.plot([qq], [v], 'o', ms=6, color=C[0], mec='white', mew=1.5)
        a.annotate(f'{v:.1e}', (qq, v), xytext=(10, 8), textcoords='offset points', fontsize=7.5)
    a.set(xlabel='assumed width / period, $W/a_P$', ylabel='$\\tau_P^{\\rm ansatz}/\\tau_{\\rm amp}$')
    g = np.linspace(0.02, 0.2, 100)
    for k, (b1, lab) in enumerate([(0.4456, '$\\beta_1$ = 0.4456 (Eshelby)'), (1.0, '$\\beta_1$ = 1 (effective)')]):
        y = (np.sqrt(3) - 1) * b1 * g ** 2 * G_CU * B_CU / 3 * 1e3
        b.plot(g, y, color=C[k], ls=LS[k], label=lab)
        yv = (np.sqrt(3) - 1) * b1 * 0.137 ** 2 * G_CU * B_CU / 3 * 1e3
        b.plot([0.137], [yv], 'o', ms=6, color=C[k], mec='white', mew=1.5, zorder=5)
        b.annotate(f'{yv:.1f}', (0.137, yv), xytext=((-34, 6) if k else (8, -12)), textcoords='offset points', fontsize=7.5)
    b.axhline(45, color=MUTED, lw=1.0, ls='--')
    b.text(0.024, 48.5, 'Cu, exp.: 45', fontsize=7, color=MUTED)
    b.set(xlabel='shear eigenstrain $\\gamma_0$', ylabel='$\\gamma_{\\rm est}$ (mJ m$^{-2}$)')
    b.legend(loc='upper left', fontsize=7)
    L = np.logspace(-7, -5, 100)
    tau = G_CU * B_CU / (2 * np.pi * (1 - NU) * L) / 1e6
    c.axvspan(1, 10, color=GRID, alpha=0.8, lw=0)
    c.text(3.2, 12, 'superplastic\ngrain sizes,\n$L\\approx d$', ha='center', fontsize=7, color=MUTED)
    c.loglog(L * 1e6, tau, color=C[0])
    t1 = G_CU * B_CU / (2 * np.pi * (1 - NU) * 1e-6) / 1e6
    c.plot([1], [t1], 'o', ms=6, color=C[0], mec='white', mew=1.5)
    c.annotate(f'{t1:.2f} MPa', (1, t1), xytext=(-46, -14), textcoords='offset points', fontsize=7.5)
    c.set(xlabel='segment length L (μm)', ylabel='$\\tau_{\\rm FR}$ (MPa)')
    for ax, l in zip(axs, 'abc'):
        panel(ax, l)
    save(fig, 'fig_conditional_estimates.pdf')


def fig_read_shockley():
    E0 = G_CU * B_CU / (4 * np.pi * (1 - NU))
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    th = np.linspace(0.05, 45, 400)
    for k, thm in enumerate([10, 15, 20]):
        tm = np.radians(thm)
        t = np.radians(th)
        g = np.where(th <= thm, E0 * t * (1 + np.log(tm) - np.log(t)), E0 * tm)
        lo = th <= thm
        ax.plot(th[lo], g[lo], color=C[k], ls='-', label=f'$\\theta_m$ = {thm}° (supplied)')
        ax.plot(th[~lo], g[~lo], color=C[k], ls=':', lw=1.5)
    ax.text(44, 0.32, 'dotted: empirical saturation,\nnot derived', ha='right', va='center', fontsize=7, color=MUTED)
    ax.set(xlabel='tilt misorientation θ (deg)', ylabel='$\\gamma_{GB}$ (J m$^{-2}$)', xlim=(0, 45), ylim=(0, 0.6))
    ax.legend(loc='lower right')
    save(fig, 'fig_read_shockley_scope.pdf')
    return E0


def fig_hp2025():
    rows = json.loads(ROWS.read_text())
    v = np.array([r['values'] for r in rows], float)
    g0, e0, E, Q, gf, ge = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
    a, b, c, d = axs.ravel()
    a.hist(g0, bins=np.arange(0.03, 0.16, 0.01), color=C[0], edgecolor='white', linewidth=1.5)
    a.set_ylim(0, 16.5)
    a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for xv, lab, k, yl, ha, dx in ((g0.mean(), f'mean {g0.mean():.4f}', 0, 15.3, 'left', 0.002),
                                   (0.05, 'model 0.05', 1, 15.3, 'right', -0.002),
                                   (0.06, 'model 0.06', 1, 13.3, 'left', 0.002)):
        a.axvline(xv, color=INK if k == 0 else C[1], ls='-' if k == 0 else '--', lw=1.2)
        a.text(xv + dx, yl, lab, fontsize=7, ha=ha)
    a.set(xlabel='fitted (tensor) shear strain $\\gamma_0$', ylabel='systems')
    gg = np.linspace(0.03, 0.15, 10)
    b.plot(gg, gg / np.sqrt(3), color=MUTED, ls='--', lw=1.2, label='source constraint $\\varepsilon_0=\\gamma_0/\\sqrt{3}$')
    b.scatter(g0, e0, s=26, color=C[0], edgecolor='white', linewidth=1.0, zorder=3, label='41 transcribed rows')
    b.set(xlabel='fitted $\\gamma_0$', ylabel='fitted $\\varepsilon_0$')
    b.legend(loc='upper left', fontsize=7)
    r = np.corrcoef(E, Q)[0, 1]
    ee = np.linspace(0, E.max() * 1.05, 10)
    c.plot(ee, ee, color=MUTED, ls='--', lw=1.2, label='Q = $E_{\\rm el}$')
    m = (Q / E).mean()
    c.plot(ee, m * ee, color=C[1], ls='-.', lw=1.2, label=f'Q = {m:.2f} $E_{{\\rm el}}$ (mean ratio)')
    c.scatter(E, Q, s=26, color=C[0], edgecolor='white', linewidth=1.0, zorder=3)
    c.set(xlabel='author elastic estimate $E_{\\rm el}$ (eV)', ylabel='source $Q$ (eV)')
    c.text(0.97, 0.05, f'Pearson r = {r:.2f}', transform=c.transAxes, ha='right', fontsize=7.5)
    c.legend(loc='upper left', fontsize=7)
    ok = (gf > 0) & (ge > 0)
    lim = [min(gf[ok].min(), ge[ok].min()) * 0.5, max(gf[ok].max(), ge[ok].max()) * 2]
    xx = np.logspace(np.log10(lim[0]), np.log10(lim[1]), 10)
    d.fill_between(xx, xx / 3, xx * 3, color=GRID, alpha=0.8, lw=0, label='within ×3')
    d.loglog(xx, xx, color=MUTED, ls='--', lw=1.2, label='1:1')
    d.scatter(ge[ok], gf[ok], s=26, color=C[0], edgecolor='white', linewidth=1.0, zorder=3)
    ratio = gf[ok] / ge[ok]
    d.text(0.03, 0.60, f'ratio median {np.median(ratio):.2f}\nrange {ratio.min():.2f}–{ratio.max():.1f}',
           transform=d.transAxes, ha='left', va='top', fontsize=7)
    d.set(xlabel='author elastic estimate $\\gamma_B^{\\rm est}$ (J m$^{-2}$)',
          ylabel='fitted $\\gamma_B^{\\rm fit}$ (J m$^{-2}$)', xlim=lim, ylim=lim)
    d.legend(loc='lower right', fontsize=7)
    for ax, l in zip(axs.ravel(), 'abcd'):
        panel(ax, l)
    save(fig, 'fig_hp2025_comparison.pdf')
    return dict(g0_mean=g0.mean(), e0_mean=e0.mean(), QE_mean=m, QE_r=r,
                ratio_median=float(np.median(ratio)), ratio_min=float(ratio.min()),
                ratio_max=float(ratio.max()), n=len(rows))


def fig_audit():
    th = np.linspace(0, 90, 361)
    T = (1 + NU) / 2 - 15 / 8 * np.sin(np.radians(2 * th)) ** 2
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot(th, T, color=C[0])
    for xv, yv, lab, xy in ((0, 2 / 3, '2/3 on axis', (22, 0.55)), (45, -29 / 24, '−29/24 on diagonal', (56, -1.27))):
        ax.plot([xv], [yv], 'o', ms=6, color=C[0], mec='white', mew=1.5, zorder=5)
        ax.annotate(lab, (xv, yv), xytext=xy, textcoords='data', fontsize=7.5,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8, shrinkB=4))
    ax.set(xlabel='direction θ in the $x_1$–$x_3$ plane (deg)', ylabel='angular factor $\\mathcal{T}_{13}$',
           xlim=(0, 90), ylim=(-1.38, 0.85), xticks=[0, 15, 30, 45, 60, 75, 90])
    save(fig, 'fig_angular_factor.pdf')
    x = np.linspace(-6, 6, 600)
    lor = lambda c: 1 / (np.pi * (1 + (x - c) ** 2))
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.plot(x, 3 * lor(0), color=C[0], ls=LS[0], label='three coincident centres')
    ax.plot(x, lor(-2) + lor(0) + lor(2), color=C[1], ls=LS[1], label='centres at $x/W=-2, 0, 2$')
    ax.set(xlabel='$x/W$', ylabel='slip density / $(b_{\\rm eff}/W)$', xlim=(-6, 6), ylim=(0, 1.15))
    ax.legend(loc='upper right', fontsize=7)
    save(fig, 'fig_slip_profiles.pdf')


def fig_schematic():
    from matplotlib.patches import Ellipse, FancyArrowPatch
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.35))
    # (a) geometry and eigenstrain; generous gaps between every label and line
    a.add_patch(Ellipse((0, 0), 2, 1, facecolor='#cde2fb', edgecolor=C[0], lw=2))
    for y, sgn in ((0.5, 1), (-0.5, -1)):
        a.add_patch(FancyArrowPatch((-0.35 * sgn, y + 0.13 * sgn), (0.35 * sgn, y + 0.13 * sgn),
                                    arrowstyle='-|>', mutation_scale=12, color=C[1], lw=2))
    a.text(0, 0.98, 'shear eigenstrain\n$\\varepsilon^*_{13}=\\gamma_0/2$', ha='center', va='bottom',
           fontsize=8.5, linespacing=1.4)
    a.text(0, 0, 'dilatation\n$\\varepsilon^*_{kk}=\\varepsilon_0$', ha='center', va='center',
           fontsize=8.5, linespacing=1.4)
    a.annotate('', (-1, -1.05), (1, -1.05), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=1))
    a.text(0, -1.2, '$2W$', ha='center', va='top', fontsize=9)
    a.annotate('', (1.3, -0.5), (1.3, 0.5), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=1))
    a.text(1.42, 0, '$2\\alpha W$', va='center', fontsize=9)
    ox, oy = -1.85, 0.75                       # coordinate triad, upper left, clear of the body
    for (dx, dy, lab, ha, va) in ((0.42, 0, '$x_1$', 'left', 'center'), (0, 0.42, '$x_3$', 'center', 'bottom')):
        a.annotate('', (ox + dx, oy + dy), (ox, oy), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1))
        a.text(ox + dx * 1.12 + (0.03 if ha == 'left' else 0), oy + dy * 1.12 + (0.03 if va == 'bottom' else 0),
               lab, fontsize=9, ha=ha, va=va)
    a.set(xlim=(-2.0, 1.95), ylim=(-1.55, 1.75), aspect='equal')
    a.axis('off')
    # (b) angular factor of the leading shear stress in the x1-x3 plane
    cmap = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#f0efec', '#e34948'])
    x = np.linspace(-4, 4, 900)
    X, Z = np.meshgrid(x, x)
    R = np.hypot(X, Z)
    th = np.arctan2(Z, X)
    T = (1 + NU) / 2 - 15 / 8 * np.sin(2 * th) ** 2
    F = np.where((R > 1.2) & (R < 4), T, np.nan)
    im = b.pcolormesh(X, Z, F, cmap=cmap, norm=TwoSlopeNorm(0, -29 / 24, 2 / 3), shading='auto', rasterized=True)
    b.contour(X, Z, np.where(R > 1.2, T, np.nan), levels=[0], colors=[MUTED], linewidths=0.8, linestyles='--')
    b.add_patch(Ellipse((0, 0), 2, 1, facecolor='#cde2fb', edgecolor=C[0], lw=1.5))
    b.set(xlabel='$x_1/W$', ylabel='$x_3/W$', aspect='equal', xlim=(-4, 4), ylim=(-4, 4),
          xticks=[-4, -2, 0, 2, 4], yticks=[-4, -2, 0, 2, 4])
    b.grid(False)
    cb = fig.colorbar(im, ax=b, fraction=0.046, pad=0.06, ticks=[-1.0, -0.5, 0, 0.5])
    cb.set_label('angular factor $\\mathcal{T}_{13}$', fontsize=9, labelpad=6)
    b.text(0, 2.9, '+', ha='center', va='center', fontsize=13, color=INK)
    b.text(2.05, 2.05, '−', ha='center', va='center', fontsize=15, color='white')
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.14, top=0.90)
    fig.text(0.02, 0.95, '(a)', fontweight='bold', va='top')
    bb = b.get_position()
    fig.text(bb.x0 - 0.07, 0.95, '(b)', fontweight='bold', va='top')
    fig.savefig(OUT / 'fig_ostz_schematic.pdf', dpi=600)
    # The graphical abstract is Figure 1 itself.
    fig.savefig(OUT / 'graphical_abstract.pdf', dpi=600)
    fig.savefig(OUT / 'graphical_abstract.png', dpi=600)
    plt.close(fig)
    print('wrote fig_ostz_schematic.pdf and graphical_abstract.pdf/.png')


def fig_strip():
    r = np.logspace(-0.3, 3, 400)               # distance from the strip edge, units of W
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.loglog(r, 1 / r, color=MUTED, ls='--', lw=1.2, label='straight dislocation, $1/r$')
    for k, D in enumerate([5, 50, 500]):
        ax.loglog(r, 1 / r - 1 / (r + D), color=C[k], ls=LS[k], label=f'slipped strip, $D$ = {D}$W$')
        ax.plot([D], [1 / D - 1 / (2 * D)], 'o', ms=5, color=C[k], mec='white', mew=1.2, zorder=5)
    ax.text(0.6, 3e-7, 'far from the strip: $D/r^2$\n($r^{-3}$ for a bounded loop)', fontsize=7, color=MUTED)
    ax.text(0.6, 2e-5, 'near the edge ($r\\ll D$): $\\approx 1/r$', fontsize=7, color=MUTED)
    ax.set(xlabel='distance from the edge, r / W', ylabel='stress ahead of the edge (normalised)',
           ylim=(1e-7, 3))
    ax.legend(loc='upper right', fontsize=7)
    save(fig, 'fig_slipped_strip.pdf')


if __name__ == '__main__':
    fig_strip()
    fig_schematic()
    fig_audit()
    s0 = fig_s1313()
    fig_decay()
    fig_kinetics()
    fig_meanfield()
    fig_estimates()
    E0 = fig_read_shockley()
    stats = fig_hp2025()
    print(json.dumps({'S1313': s0, 'beta1': 1 - 2 * s0, 'E0_Cu_J_m2': E0, **stats}, indent=1))
