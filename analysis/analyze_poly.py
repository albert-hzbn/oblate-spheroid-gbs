#!/usr/bin/env python3
"""Analyse the nanocrystalline Cu tension runs.

For every run directory: flow stress (mean tensile stress over 6-10 % strain),
peak stress, and, on the snapshot nearest 10 % strain, DXA line lengths by
Burgers-vector family (Shockley partial 1/6<112>, perfect 1/2<110>, other) and
the adaptive-CNA fraction of HCP atoms (stacking faults and twins).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from ovito.io import import_file
from ovito.modifiers import (CommonNeighborAnalysisModifier,
                             DislocationAnalysisModifier)

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'md' / 'polycrystal' / 'runs'   # simulation output (dumps needed for DXA)
OUT = REPO / 'results'


def stress(run):
    d = np.loadtxt(run / 'stress.dat', skiprows=1)
    e, s = d[:, 1], d[:, 2]
    m = (e >= 0.06) & (e <= 0.10)
    return dict(flow_GPa=float(s[m].mean()) if m.any() else None,
                flow_sd_GPa=float(s[m].std()) if m.any() else None,
                peak_GPa=float(s.max()), peak_strain=float(e[s.argmax()]),
                final_strain=float(e[-1]))


def last_dump(run):
    ds = sorted(run.glob('dump.*.lmp'), key=lambda p: int(re.findall(r'\d+', p.name)[-1]))
    return ds[-1] if ds else None


def structure(path):
    pipe = import_file(str(path))
    pipe.modifiers.append(CommonNeighborAnalysisModifier())
    pipe.modifiers.append(DislocationAnalysisModifier(
        input_crystal_structure=DislocationAnalysisModifier.Lattice.FCC))
    d = pipe.compute()
    st = np.asarray(d.particles['Structure Type'])
    n = len(st)
    frac = {name: float((st == val).sum() / n) for name, val in
            (('fcc', 1), ('hcp', 2), ('other', 0))}
    lengths = {'shockley': 0., 'perfect': 0., 'other': 0.}
    for seg in d.dislocations.segments:
        bv = np.sort(np.abs(np.asarray(seg.true_burgers_vector, dtype=float)))
        mag = np.linalg.norm(bv)
        if np.allclose(bv, [1/6, 1/6, 2/6], atol=1e-3):
            lengths['shockley'] += seg.length
        elif np.allclose(bv, [0, 0.5, 0.5], atol=1e-3):
            lengths['perfect'] += seg.length
        else:
            lengths['other'] += seg.length
    vol = float(abs(np.linalg.det(np.asarray(d.cell)[:, :3])))
    dens = {k + '_density_m2': v / vol * 1e20 for k, v in lengths.items()}
    return dict(dump=path.name, n_atoms=n, **frac, **{k + '_length_A': v for k, v in lengths.items()},
                **dens, volume_A3=vol)


def main():
    runs = sorted(p for p in ROOT.iterdir() if (p / 'stress.dat').exists())
    if len(sys.argv) > 1:
        runs = [r for r in runs if any(a in r.name for a in sys.argv[1:])]
    res = {}
    for r in runs:
        row = dict(run=r.name, **stress(r))
        ld = last_dump(r)
        if ld is not None and (r / 'final.data').exists():
            row.update(structure(ld))
        res[r.name] = row
        print(r.name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                       if k in ('flow_GPa', 'peak_GPa', 'hcp', 'shockley_density_m2', 'perfect_density_m2')}, flush=True)
    (OUT / 'poly_summary.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
