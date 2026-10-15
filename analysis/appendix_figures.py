#!/usr/bin/env python3
"""Figures for the Appendix (proposed extension), from the md_appendix calculations."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
R = REPO / 'results'
OUT = REPO / 'figures' / 'main'
OUT.mkdir(exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dd'
LS = ['-', '--', '-.', ':', (0, (5, 1, 1, 1))]
MK = ['o', 's', '^', 'D', 'v']
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 2.0, 'legend.frameon': False, 'pdf.fonttype': 42})


def panel(ax, l):
    ax.text(-0.17, 1.04, f'({l})', transform=ax.transAxes, fontweight='bold', va='bottom')


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name); plt.close(fig); print('wrote', name)


# A1: GSFE ------------------------------------------------------------------
def fig_gsfe():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bp = 3.615 / 6 ** 0.5
    for k, (f, lab, b) in enumerate((('gsfe_partial.dat', 'partial path, $[11\\bar2]$', bp),
                                      ('gsfe_full.dat', 'full path, $[\\bar110]$', 3.615 / 2 ** 0.5))):
        d = np.loadtxt(R / 'gsfe' / f, skiprows=1)
        ax.plot(d[:, 1] / bp, d[:, 2], color=C[k], ls=LS[k], label=lab)
    d = np.loadtxt(R / 'gsfe' / 'gsfe_partial.dat', skiprows=1)
    i = np.argmax(d[:, 2])
    ax.plot([d[i, 1] / bp], [d[i, 2]], 'o', ms=6, color=C[0], mec='white', mew=1.5, zorder=5)
    ax.annotate(f'$\\gamma_{{usf}}$ = {d[i, 2]:.0f}', (d[i, 1] / bp, d[i, 2]), xytext=(6, 10),
                textcoords='offset points', fontsize=7.5)
    gs = np.interp(bp, d[:, 1], d[:, 2])
    ax.plot([1], [gs], 'o', ms=6, color=C[0], mec='white', mew=1.5, zorder=5)
    ax.annotate(f'$\\gamma_{{isf}}$ = {gs:.1f}', (1, gs), xytext=(1.26, 70), textcoords='data', fontsize=7.5,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8, shrinkA=2, shrinkB=4))
    ax.set(xlabel='displacement / $b_p$', ylabel='energy (mJ m$^{-2}$)', xlim=(0, 1.75), ylim=(0, 520))
    ax.legend(loc='upper left')
    save(fig, 'figA1_gsfe.pdf')


# A2: grain-size series --------------------------------------------------------
def fig_grain():
    d = json.load(open(R / 'poly_summary.json'))
    fit = json.load(open(R / 'ihp_fit.json'))
    rows = {}
    for k, r in d.items():
        if k.startswith('rate'):
            continue
        D = float(k.split('_')[0][1:].replace('s2', ''))
        s, p = r['shockley_density_m2'], r['perfect_density_m2']
        rows.setdefault(D, []).append((r['flow_GPa'], s / (s + p)))
    Ds = np.array(sorted(rows))
    fl = [np.array([x[0] for x in rows[D]]) for D in Ds]
    pf = [np.array([x[1] for x in rows[D]]) for D in Ds]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for D, v in zip(Ds, fl):
        a.plot([D] * len(v), v, 'o', ms=4, color=C[0], alpha=0.45, mec='none')
    a.plot(Ds, [v.mean() for v in fl], 'o', ms=6, color=C[0], mec='white', mew=1.5, zorder=5, label='MD, mean of seeds')
    x = np.linspace(2.5, 17, 300)
    f = lambda x, A, d0, B: A * np.sqrt(np.clip(1 - d0 / x, 0, None) / x) + B
    a.plot(x, f(x, fit['A'], fit['d0'], fit['B']), color=C[1], ls='--', lw=1.6,
           label=f"Eq. (3) form, $d_0$ = {fit['d0']:.1f} nm")
    a.plot(x, f(x, fit['fixed']['A'], 3.13, fit['fixed']['B']), color=C[2], ls=':', lw=1.6,
           label='Eq. (3) form, $d_0$ = 3.13 nm ($W=2.5a_0$)')
    a.set(xlabel='grain size $d$ (nm)', ylabel='flow stress (GPa)', ylim=(0.2, 2.6))
    a.legend(loc='lower right', fontsize=6.8, handlelength=1.6)
    for D, v in zip(Ds, pf):
        b.plot([D] * len(v), v, 'o', ms=4, color=C[0], alpha=0.45, mec='none')
    b.plot(Ds, [v.mean() for v in pf], 'o-', ms=6, color=C[0], mec='white', mew=1.5, lw=1.2)
    b.set(xlabel='grain size $d$ (nm)', ylabel='Shockley fraction of line length', ylim=(0.6, 1.0))
    panel(a, 'a'); panel(b, 'b')
    save(fig, 'figA2_grain_size.pdf')


# A3: rate and temperature ------------------------------------------------------
def fig_rate():
    rs = json.load(open(R / 'rate_summary.json'))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for k, (T, v) in enumerate(sorted(rs['per_T'].items(), key=lambda t: float(t[0]))):
        r = np.array([p['rate'] for p in v['points']]); s = np.array([p['sigma_GPa'] for p in v['points']])
        a.semilogx(r, s, MK[k], ms=6, color=C[k], mec='white', mew=1.2, ls='none')
        c = np.polyfit(np.log(r), s, 1)
        rr = np.logspace(np.log10(r.min()), np.log10(r.max()), 50)
        a.semilogx(rr, np.polyval(c, np.log(rr)), color=C[k], ls=LS[k], lw=1.4,
                   label=f"{float(T):.0f} K, $V^*$ = {v['V_star_b3']:.1f}$b^3$")
    a.set(xlabel='strain rate (s$^{-1}$)', ylabel='flow stress (GPa)', ylim=(1.0, 2.95))
    a.legend(loc='upper left')
    q = rs['Q_at_sigma']
    b.plot(q['sigma_GPa'], q['Q_eV'], 'o-', ms=6, color=C[0], mec='white', mew=1.5)
    c = np.polyfit(q['sigma_GPa'], q['Q_eV'], 1)
    xs = np.linspace(0, 1.7, 50)
    b.plot(xs, np.polyval(c, xs), color=MUTED, ls='--', lw=1.0)
    b.plot([0], [c[1]], 'o', ms=6, color=C[1], mec='white', mew=1.5, zorder=5)
    b.annotate(f'extrapolated: {c[1]:.2f} eV', (0, c[1]), xytext=(0.12, 0.45), textcoords='data', fontsize=7.5, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8, shrinkB=4))
    b.axhline(0.38, color=C[2], ls=':', lw=1.2)
    b.text(1.68, 0.385, 'model value 0.38 eV', ha='right', va='bottom', fontsize=7, color=MUTED)
    b.set(xlabel='flow stress (GPa)', ylabel='apparent activation energy (eV)', xlim=(0, 1.7), ylim=(0, 0.5))
    panel(a, 'a'); panel(b, 'b')
    save(fig, 'figA3_activation.pdf')
    return c


# A4: quasi-static emission for a set of boundaries, and converged barriers vs stress -------
def fig_emission():
    from scipy.optimize import curve_fit
    N = R
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    sets = [('c0_s30 reverse', N / 'bicrystal_quasistatic' / 'c0_s30_reverse' / 'qs.dat', C[0], '-'),
            ('c0_s03 reverse', N / 'bicrystal_quasistatic' / 'c0_s03_reverse' / 'qs.dat', C[1], '--'),
            ('cp_s21 reverse', N / 'bicrystal_quasistatic' / 'cp_s21_reverse' / 'qs.dat', C[2], '-.'),
            ('c0_s30 positive', N / 'bicrystal_quasistatic' / 'c0_s30_positive' / 'qs.dat', MUTED, ':'),
            ('Σ7 reverse', N / 'bicrystal_quasistatic' / 'sigma7_reverse' / 'qs.dat', C[3], (0, (5, 1, 1, 1)))]
    for lab, f, col, ls in sets:
        q = np.genfromtxt(f, names=True)
        m = np.abs(q['strain']) <= 0.16
        a.plot(np.abs(q['strain'])[m], np.abs(q['tau_yz_GPa'])[m], color=col, ls=ls, lw=1.5, label=lab)
    a.set(xlabel='applied shear strain', ylabel='|shear stress| (GPa)', xlim=(0, 0.16), ylim=(0, 2.7))
    a.legend(loc='upper left', fontsize=6.5, ncol=2, columnspacing=1.0, handlelength=1.8)
    qs = np.genfromtxt(N / 'bicrystal_quasistatic' / 'c0_s30_reverse' / 'qs.dat', names=True)
    tau_of = {int(k): -t for k, t in zip(qs['k'], qs['tau_yz_GPa'])}
    def barrier(f, n):
        lines = [l for l in open(f) if l.strip() and l.split()[0].isdigit()]
        x = lines[-1].split()
        pe = np.array([float(x[10 + 2 * i]) for i in range(n)])
        return float(np.max(pe - pe[0]))
    kd = [20, 19, 18, 17]
    td = np.array([tau_of[k] for k in kd]); qd = np.array([barrier(N / 'neb' / f'dense_k{k}' / 'log.master', 16) for k in kd])
    k8 = [20, 18]
    t8 = np.array([tau_of[k] for k in k8]); q8 = np.array([barrier(N / 'neb' / f'emit_k{k}' / 'log.master', 8) for k in k8])
    f = lambda t, A, tc, al: A * np.clip(1 - t / tc, 1e-9, None) ** al
    p, _ = curve_fit(f, td, qd, p0=[100, 1.1, 1.5], bounds=([1, 1.079, 0.5], [5000, 1.5, 4]))
    tt = np.linspace(0.68, p[1], 200)
    b.plot(tt, f(tt, *p), color=C[0], lw=1.4, ls='--',
           label=f'fit $A(1-\\tau/\\tau_c)^{{\\alpha}}$: $\\tau_c$ = {p[1]:.2f} GPa, $\\alpha$ = {p[2]:.2f}')
    b.plot(td, qd, 'o', ms=7, color=C[0], mec='white', mew=1.5, label='16 images', zorder=5)
    b.plot(t8, q8, 's', ms=9, mfc='none', mec=C[1], mew=1.6, label='8 images', zorder=6)
    b.set(xlabel='shear stress (GPa)', ylabel='emission barrier (eV)', xlim=(0.68, 1.14), ylim=(0, 28))
    b.legend(loc='upper right', fontsize=6.8)
    panel(a, 'a'); panel(b, 'b')
    save(fig, 'figA4_emission.pdf')
    return dict(tau=td.tolist(), Q=qd.tolist(), fit=dict(A=p[0], tau_c=p[1], alpha=p[2]),
                V_local_b3=[float(-(qd[i + 1] - qd[i]) / (td[i + 1] - td[i]) * 1.602e-19 / 1e9 / (0.2556e-9) ** 3)
                            for i in range(3)])


# A5: rate law with the unit activation energy vs all digitised experimental points ------------
def fig_ensemble():
    f = json.load(open(R / 'mesoscale' / 'fullset_comparison.json'))
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    lim = (2e-6, 3e-3)
    x = np.array(lim)
    ax.fill_between(x, x / 10, x * 10, color=GRID, alpha=0.5, lw=0, label='within ×10')
    ax.fill_between(x, x / 2, x * 2, color=GRID, alpha=1.0, lw=0, label='within ×2')
    ax.loglog(x, x, color=MUTED, ls='--', lw=1.0)
    conds = sorted({(r['d_um'], r['T']) for r in f['rows']})
    for k, c in enumerate(conds):
        rr = [r for r in f['rows'] if (r['d_um'], r['T']) == c]
        ax.loglog([r['expt'] for r in rr], [r['esh_1e13'] for r in rr], ['o', 's', '^', 'D', 'v', 'P', 'X'][k],
                  ms=5.5, color=(C + [INK, '#7a5195'])[k], mec='white', mew=0.8, ls='none',
                  label=f'{c[1]} K, {c[0]} µm')
    st = f['eshelby_1e13']
    ax.text(0.97, 0.04, f"{st['n']} points: median ratio {st['median_ratio']:.2f};\n{st['within_x2']} within ×2, {st['within_x10']} within ×10",
            transform=ax.transAxes, ha='right', fontsize=7)
    ax.set(xlabel='measured shear strain rate (s$^{-1}$)', ylabel='predicted shear strain rate (s$^{-1}$)',
           xlim=lim, ylim=lim)
    ax.legend(loc='upper left', fontsize=6.3, ncol=2, columnspacing=0.8, handlelength=1.2)
    save(fig, 'figA5_ensemble.pdf')


if __name__ == '__main__':
    fig_gsfe(); fig_grain(); c = fig_rate(); em = fig_emission(); fig_ensemble()
    print('emission', json.dumps(em))
    print('Q(sigma) linear fit: intercept %.3f eV, slope %.3f eV/GPa' % (c[1], c[0]))
