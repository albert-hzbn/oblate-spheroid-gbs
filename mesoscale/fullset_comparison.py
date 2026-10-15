#!/usr/bin/env python3
"""Rate law of Padmanabhan and Divinski (2024), Eq. (2), at all digitised experimental points.

Shear stresses from Fig. A1 (digitised); per-condition parameters from Table A1 (gamma0,
tau0, dF0); their predicted rates from Tables A2/A3 are used to check the reading. The
present analysis replaces only dF0 by the Eshelby energy of the unit with the fitted
gamma0 read as a tensor shear strain (as confirmed by K.A. Padmanabhan, 13 Oct 2026).
"""
import json
from pathlib import Path
import numpy as np

kB, h, NA = 1.380649e-23, 6.62607015e-34, 6.02214076e23
a0 = 0.2556e-9
W = 2.5 * a0
V0 = 4 / 3 * np.pi * (2.5 * a0) ** 2 * (1.25 * a0)
G = 24e9
B1, B2 = 0.445599, 0.888889
A1 = {(7, 723): (.1443, 12.71, 174.23), (7, 773): (.1454, 9.32, 176.41), (7, 823): (.1465, 5.18, 178.75),
      (7, 873): (.1472, 3.21, 180.12), (3, 673): (.1387, 11.41, 162.95), (3, 723): (.1398, 8.25, 165.21),
      (3, 773): (.1420, 6.33, 169.52)}
PRED = {(7, 723): [5.24e-6, 1.34e-5, 1.26e-5, 2.33e-5, 2.49e-5, 4.61e-5, 4.43e-5, 7.48e-5],
        (7, 773): [4.66e-6, 1.24e-5, 1.28e-5, 2.61e-5, 2.47e-5, 4.31e-5, 1.23e-4, 1.15e-4, 2.44e-4],
        (7, 823): [5.24e-6, 5.39e-6, 1.29e-5, 2.53e-5, 2.54e-5, 5.24e-5, 1.25e-4, 2.53e-4, 5.01e-4],
        (7, 873): [4.91e-6, 1.35e-5, 2.48e-5, 5.15e-5, 1.26e-4, 2.74e-4, 5.35e-4, 1.41e-3],
        (3, 673): [4.89e-6, 1.03e-5, 2.05e-5, 4.61e-5],
        (3, 723): [4.46e-6, 1.16e-5, 1.11e-5, 2.35e-5, 2.20e-5, 4.38e-5, 4.73e-5, 2.40e-4],
        (3, 773): [4.89e-6, 1.10e-5, 2.38e-5, 2.31e-5, 4.86e-5, 4.93e-5, 1.19e-4, 2.36e-4, 2.54e-4, 5.56e-4]}


def rate(tau, tau0, g0, F, T, d, nu):
    return 2 * W * g0 * nu / d * np.sinh((tau - tau0) * g0 * V0 / (2 * kB * T)) * np.exp(-F / (kB * T))


dig = json.load(open(Path(__file__).resolve().parents[1] / 'results' / 'mesoscale' / 'kap2024_figA1_digitized.json'))
rows = []
for key, v in dig.items():
    d, T = v['d_um'], v['T']
    g0, t0, F = A1[(d, T)]
    e0 = g0 / np.sqrt(3)
    F_fit = F * 1e3 / NA
    F_esh = 0.5 * (B1 * (2 * g0) ** 2 + B2 * e0 ** 2) * G * V0
    for i, r in enumerate(v['rows']):
        if r['tau_MPa'] is None:
            continue
        tau = r['tau_MPa'] * 1e6
        rows.append(dict(d_um=d, T=T, tau_MPa=r['tau_MPa'], expt=r['expt_rate'], kap_pred=PRED[(d, T)][i],
                         fit_1e13=rate(tau, t0 * 1e6, g0, F_fit, T, d * 1e-6, 1e13),
                         esh_1e13=rate(tau, t0 * 1e6, g0, F_esh, T, d * 1e-6, 1e13),
                         esh_kTh=rate(tau, t0 * 1e6, g0, F_esh, T, d * 1e-6, kB * T / h),
                         dF0_fit=F, dF0_esh=F_esh * NA / 1e3))


def stats(key):
    lr = np.array([np.log10(r[key] / r['expt']) for r in rows])
    return dict(median_ratio=float(10 ** np.median(lr)), within_x2=int((np.abs(lr) <= np.log10(2)).sum()),
                within_x10=int((np.abs(lr) <= 1).sum()), n=len(lr), rms_log10=float(np.sqrt(np.mean(lr ** 2))))


res = dict(n_points=len(rows), check_fit_vs_published=stats('fit_1e13'),
           published_pred=stats('kap_pred'), eshelby_1e13=stats('esh_1e13'), eshelby_kTh=stats('esh_kTh'),
           reading_check=float(np.median([r['fit_1e13'] / r['kap_pred'] for r in rows])), rows=rows)
for k in ('published_pred', 'check_fit_vs_published', 'eshelby_1e13', 'eshelby_kTh'):
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in res[k].items()})
print('median (our evaluation of their fit) / (their published prediction):', round(res['reading_check'], 3))
by = {}
for r in rows:
    by.setdefault((r['d_um'], r['T']), []).append(np.log10(r['esh_1e13'] / r['expt']))
for k in sorted(by):
    print(k, 'Eshelby/expt median ratio %.2f' % 10 ** np.median(by[k]))
(Path(__file__).resolve().parents[1] / 'results' / 'mesoscale' / 'fullset_comparison.json').write_text(json.dumps(res, indent=1) + '\n')
