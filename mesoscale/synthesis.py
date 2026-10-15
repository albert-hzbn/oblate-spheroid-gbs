#!/usr/bin/env python3
"""Synthesis of the single-unit analysis with the mesoscopic boundary-sliding model.

Uses the governing equations of Padmanabhan and Divinski (2024, Mater. Sci. Eng. A 908,
146713), Eqs. (2), (3), (A.3), and their Cu-alloy fits (Table A1; lowest-rate rows of
Tables A2, A3):
  rate     gdot = (2 W g0 nu / d) sinh[(tau - tau0) g0 V0 / (2kT)] exp(-dF0/kT)
  energy   dF0  = 1/2 (b1 g0^2 + b2 e0^2) G V0
  thresh.  tau0 = G [ (d1/d)(1 - d0/d) ]^0.5,  d0 = 2 sqrt(6) W,
           d1 = 2^1.5 3^-0.75 N^-0.5 gB/(G af)
(the printed d/d1 is taken as d1/d: only this form reproduces their Eq. (4) prefactor).
"""
import json
from pathlib import Path
import numpy as np

kB, h, NA, eV = 1.380649e-23, 6.62607015e-34, 6.02214076e23, 1.602176634e-19
a0 = 0.2556e-9                      # Cu atomic diameter (m)
W = 2.5 * a0                        # boundary thickness = spheroid thickness (m)
V0 = 4 / 3 * np.pi * (2.5 * a0) ** 2 * (1.25 * a0)   # equatorial radius 2.5 a0, half-thickness 1.25 a0
G, gB, af = 24e9, 0.72, 1.0
B1_ESH, B2_ESH = 0.445599, 0.888889  # present homogeneous Eshelby coefficients (engineering shear)

# Table A1: (d um, T K, min stress MPa, g0, tau0 MPa, dF0 kJ/mol, N); expt rate of row 1 of A2/A3
A1 = [(7, 723, 31.95, .1443, 12.71, 174.23, 3.9, 5.63e-6),
      (7, 773, 17.88, .1454, 9.32, 176.41, 5.3, 5.51e-6),
      (7, 823, 10.79, .1465, 5.18, 178.75, 7.9, 5.99e-6),
      (7, 873, 8.08, .1472, 3.21, 180.12, 11.2, 5.75e-6),
      (3, 673, 28.59, .1387, 11.41, 162.95, 4.3, 5.28e-6),
      (3, 723, 18.19, .1398, 8.25, 165.21, 7.29, 5.18e-6),
      (3, 773, 13.97, .1420, 6.33, 169.52, 9.12, 5.40e-6)]


def rate(tau, tau0, g0, dF0_J, T, d, nu):
    return 2 * W * g0 * nu / d * np.sinh((tau - tau0) * g0 * V0 / (2 * kB * T)) * np.exp(-dF0_J / (kB * T))


def dF0(g0_eng, e0, b1=B1_ESH, b2=B2_ESH):
    return 0.5 * (b1 * g0_eng ** 2 + b2 * e0 ** 2) * G * V0


def tau0_eq3(d, N, Gb=G, gb=gB, Wb=W):
    d0 = 2 * np.sqrt(6) * Wb
    d1 = 2 ** 1.5 * 3 ** -0.75 * N ** -0.5 * gb / (Gb * af)
    x = (d1 / d) * (1 - d0 / d)
    return np.where(d > d0, Gb * np.sqrt(np.clip(x, 0, None)), 0.0), d0, d1


rows = []
for d_um, T, tau, g0, t0, F, N, expt in A1:
    d = d_um * 1e-6
    e0 = g0 / np.sqrt(3)
    F_fit = F * 1e3 / NA
    out = dict(d_um=d_um, T=T, stress_MPa=tau, g0_fit=g0, tau0_MPa=t0, dF0_fit_kJmol=F, N=N, expt_rate=expt)
    for nu_name, nu in (('nu1e13', 1e13), ('kT_h', kB * T / h)):
        out[f'rate_fitparams_{nu_name}'] = rate(tau * 1e6, t0 * 1e6, g0, F_fit, T, d, nu)
    # present Eshelby coefficients, two readings of the fitted g0
    for reading, g_eng in (('tensor', 2 * g0), ('engineering', g0)):
        Fp = dF0(g_eng, e0)
        out[f'dF0_eshelby_{reading}_kJmol'] = Fp * NA / 1e3
        out[f'rate_eshelby_{reading}_kT_h'] = rate(tau * 1e6, t0 * 1e6, g0, Fp, T, d, kB * T / h)
        out[f'rate_eshelby_{reading}_nu1e13'] = rate(tau * 1e6, t0 * 1e6, g0, Fp, T, d, 1e13)
    rows.append(out)

# Inverse Hall-Petch threshold, Eq. (3), nanocrystalline range, mean N of Table A1
dn = np.linspace(2.5e-9, 40e-9, 400)
Nmean = np.mean([r[6] for r in A1])
t0n, d0, d1 = tau0_eq3(dn, Nmean)
peak = dn[np.argmax(t0n)]

summary = dict(
    V0_nm3=V0 * 1e27, W_nm=W * 1e9, G_GPa=G / 1e9, gB=gB,
    rows=rows,
    mean_dF0_fit_kJmol=float(np.mean([r['dF0_fit_kJmol'] for r in rows])),
    mean_dF0_eshelby_tensor_kJmol=float(np.mean([r['dF0_eshelby_tensor_kJmol'] for r in rows])),
    mean_dF0_eshelby_engineering_kJmol=float(np.mean([r['dF0_eshelby_engineering_kJmol'] for r in rows])),
    b1_implied_by_fit=float(np.mean([(2 * r['dF0_fit_kJmol'] * 1e3 / NA / (G * V0) - B2_ESH * (r['g0_fit'] / np.sqrt(3)) ** 2)
                                      / r['g0_fit'] ** 2 for r in rows])),
    eq3=dict(N_mean=float(Nmean), d0_nm=float(d0 * 1e9), d1_nm=float(d1 * 1e9),
             peak_d_nm=float(peak * 1e9), peak_tau0_MPa=float(t0n.max() / 1e6),
             curve_d_nm=(dn * 1e9).tolist(), curve_tau0_MPa=(t0n / 1e6).tolist()))
(Path(__file__).resolve().parents[1] / 'results' / 'mesoscale' / 'synthesis.json').write_text(json.dumps(summary, indent=1) + '\n')

print(f"V0 = {V0*1e27:.4f} nm3, W = {W*1e9:.3f} nm")
print("d  T  expt | fitparams(kT/h) fitparams(1e13) | dF0 fit / esh-tensor / esh-eng (kJ/mol) | rate esh-tensor(kT/h) esh-eng(kT/h)")
for r in rows:
    print(f"{r['d_um']} {r['T']} {r['expt_rate']:.2e} | {r['rate_fitparams_kT_h']:.2e} {r['rate_fitparams_nu1e13']:.2e} | "
          f"{r['dF0_fit_kJmol']:.1f} / {r['dF0_eshelby_tensor_kJmol']:.1f} / {r['dF0_eshelby_engineering_kJmol']:.1f} | "
          f"{r['rate_eshelby_tensor_kT_h']:.2e} {r['rate_eshelby_engineering_kT_h']:.2e}")
print(f"beta1 implied by the fitted dF0 (with beta2 = {B2_ESH}): {summary['b1_implied_by_fit']:.3f}")
print(f"Eq.(3): N = {Nmean:.2f}, d0 = {d0*1e9:.2f} nm, d1 = {d1*1e9:.4f} nm, peak at {peak*1e9:.2f} nm, tau0max = {t0n.max()/1e6:.0f} MPa")
