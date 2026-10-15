#!/usr/bin/env python3
"""Activation parameters of boundary-mediated flow from the d = 5 nm rate/temperature series.

Flow stress sigma = mean tensile stress over 6-10 % strain. At each temperature the
apparent activation volume is V* = sqrt(3) kT / (d sigma / d ln rate) (von Mises
conversion to shear). The apparent activation energy at fixed stress follows from
Q = -k d ln(rate) / d(1/T) at constant sigma, using a linear sigma(ln rate) fit per T.
"""
import json
from pathlib import Path
import numpy as np

kB = 1.380649e-23
b = 0.2556e-9
REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results' / 'polycrystal'


def flow(run):
    d = np.loadtxt(run / 'stress.dat', skiprows=1)
    m = (d[:, 1] >= 0.06) & (d[:, 1] <= 0.10)
    return float(d[m, 2].mean()), float(d[m, 2].std())


data = {}
for run in sorted(ROOT.glob('rate_d5_T*_r*')):
    if not (run / 'stress.dat').exists():
        continue
    T = float(run.name.split('_T')[1].split('_')[0])
    r = float(run.name.split('_r')[1]) * 1e12          # 1/ps -> 1/s
    s, sd = flow(run)
    data.setdefault(T, []).append((r, s, sd))
# include the d = 5 nm grain-size run (300 K, 5e8 /s)
g = ROOT / 'd5_T300_r5e8'
if (g / 'stress.dat').exists():
    s, sd = flow(g)
    data.setdefault(300.0, []).append((5e8, s, sd))

res = {'per_T': {}}
fits = {}
for T, rows in sorted(data.items()):
    rows.sort()
    lr = np.log([x[0] for x in rows]); sg = np.array([x[1] for x in rows]) * 1e9
    if len(rows) >= 2:
        slope, icpt = np.polyfit(lr, sg, 1)                 # sigma = slope ln(rate) + icpt
        V = np.sqrt(3) * kB * T / slope
        fits[T] = (slope, icpt)
        res['per_T'][T] = dict(points=[dict(rate=x[0], sigma_GPa=x[1], sd_GPa=x[2]) for x in rows],
                               dsigma_dlnrate_MPa=slope / 1e6, m=float(slope / sg.mean()),
                               V_star_b3=float(V / b ** 3))
        print(f"T={T:.0f} K  dsigma/dln(rate)={slope/1e6:.1f} MPa  V*={V/b**3:.1f} b^3  m={slope/sg.mean():.3f}")
if len(fits) >= 2:
    # Q at fixed stress: ln rate(T) = (sigma - icpt)/slope
    sigs = np.linspace(min(f[1] for f in fits.values()) * 0 + 0.8e9, 1.6e9, 5)
    Qs = []
    for s in sigs:
        invT = np.array([1 / T for T in fits]); lnr = np.array([(s - fits[T][1]) / fits[T][0] for T in fits])
        Q = -kB * np.polyfit(invT, lnr, 1)[0] / 1.602176634e-19
        Qs.append(Q)
    res['Q_at_sigma'] = dict(sigma_GPa=(sigs / 1e9).tolist(), Q_eV=Qs)
    print('Q(sigma) [eV]:', ' '.join(f'{s/1e9:.2f} GPa: {q:.2f}' for s, q in zip(sigs, Qs)))
(REPO / 'results' / 'rate_summary.json').write_text(json.dumps(res, indent=1) + '\n')
