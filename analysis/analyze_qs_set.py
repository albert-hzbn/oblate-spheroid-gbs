#!/usr/bin/env python3
"""First emission instability of each boundary under 0 K quasi-static shear.

For each run in neb/qs_set (and the original c0_s30 reverse run in neb/qs), the first
instability is the first step where the shear-stress magnitude drops by more than
0.15 GPa while the HCP-atom count rises by more than 500. DXA before and after the
step distinguishes nucleation of new line length from glide of existing partials.
"""
import json
from pathlib import Path
import numpy as np
from ovito.io import import_file
from ovito.modifiers import DislocationAnalysisModifier

REPO = Path(__file__).resolve().parents[1]
R = REPO / 'md' / 'bicrystal' / 'runs'   # simulation output (dumps needed for DXA)
runs = {'c0_s30_reverse': R / 'qs'}
runs.update({p.name: p for p in sorted((R / 'qs_set').iterdir()) if (p / 'qs.dat').exists()})


def dxa(path):
    p = import_file(str(path))
    p.modifiers.append(DislocationAnalysisModifier(input_crystal_structure=DislocationAnalysisModifier.Lattice.FCC))
    d = p.compute()
    L = {'shockley': 0., 'perfect': 0., 'other': 0.}
    for s in d.dislocations.segments:
        b = np.sort(np.abs(np.asarray(s.true_burgers_vector)))
        k = 'shockley' if np.allclose(b, [1/6, 1/6, 2/6], atol=1e-3) else ('perfect' if np.allclose(b, [0, .5, .5], atol=1e-3) else 'other')
        L[k] += s.length
    return {k: round(v, 1) for k, v in L.items()}


out = {}
for name, d in runs.items():
    q = np.genfromtxt(d / 'qs.dat', names=True)
    k, tau, nh, st = q['k'].astype(int), np.abs(q['tau_yz_GPa']), q['nhcp'], np.abs(q['strain'])
    ev = None
    for i in range(1, len(k)):
        if tau[i - 1] - tau[i] > 0.15 and nh[i] - nh[i - 1] > 500:
            ev = i
            break
    row = dict(n_steps=int(len(k)), max_tau_GPa=float(tau.max()))
    if ev is not None:
        row.update(k_c=int(k[ev]), tau_c_GPa=float(tau[ev - 1]), strain_c=float(st[ev - 1]),
                   drop_GPa=float(tau[ev - 1] - tau[ev]), d_nhcp=int(nh[ev] - nh[ev - 1]))
        before, after = d / f'qs.{k[ev - 1]}.dump', d / f'qs.{k[ev]}.dump'
        if before.exists() and after.exists():
            row['dxa_before'], row['dxa_after'] = dxa(before), dxa(after)
            row['new_shockley_A'] = row['dxa_after']['shockley'] - row['dxa_before']['shockley']
    out[name] = row
    print(name, {kk: row[kk] for kk in ('tau_c_GPa', 'strain_c', 'drop_GPa', 'd_nhcp', 'new_shockley_A') if kk in row}
          if ev is not None else f"no instability (max tau {row['max_tau_GPa']:.2f} GPa)", flush=True)
REPO / 'results' / 'qs_set_summary.json'.write_text(json.dumps(out, indent=1) + '\n')
