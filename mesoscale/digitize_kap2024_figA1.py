#!/usr/bin/env python3
"""Digitise the experimental points of Padmanabhan and Divinski (2024), Fig. A1.

Axes are log-log with frames at tau = 1 and 100 MPa and rate = 1e-7 and 1e-2 /s.
Markers are found by colour, thin predicted lines (same colours) are removed by
morphological opening, and each marker is matched to an experimental rate of
Tables A2/A3 by its y position, so only the stress (x) is taken from the image.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

IMG = Path(__file__).with_name('kap2024_figA1.jpeg')   # Fig. A1 extracted from the published PDF (not redistributed)
a = np.asarray(Image.open(IMG).convert('RGB')).astype(float)
H, Wd, _ = a.shape
dark = a.sum(axis=2) < 200


def frame(x0, x1):
    """Plot frame of the panel spanning columns x0..x1: longest dark rows/columns."""
    sub = dark[:, x0:x1]
    rows = np.where(sub.sum(axis=1) > 0.6 * (x1 - x0))[0]
    cols = np.where(dark[rows.min():rows.max(), x0:x1].sum(axis=0) > 0.6 * (rows.max() - rows.min()))[0] + x0
    return cols.min(), cols.max(), rows.min(), rows.max()


panels = {'a': frame(0, Wd // 2), 'b': frame(Wd // 2, Wd)}
print('frames (left, right, top, bottom):', panels)

COL = {'a': {723: (60, 60, 60), 773: (30, 80, 220), 823: (180, 120, 220), 873: (60, 210, 210)},
       'b': {673: (60, 60, 60), 723: (40, 110, 220), 773: (180, 120, 230)}}
# experimental rates of Tables A2 (3 um) and A3 (7 um)
EXPT = {('a', 723): [5.63e-6, 1.43e-5, 1.43e-5, 2.62e-5, 2.74e-5, 5.42e-5, 5.09e-5, 8.04e-5],
        ('a', 773): [5.51e-6, 1.40e-5, 1.43e-5, 2.96e-5, 2.96e-5, 4.88e-5, 1.38e-4, 1.35e-4, 2.79e-4],
        ('a', 823): [5.99e-6, 5.89e-6, 1.43e-5, 2.91e-5, 2.91e-5, 5.65e-5, 1.35e-4, 2.86e-4, 5.54e-4],
        ('a', 873): [5.75e-6, 1.49e-5, 2.91e-5, 6.01e-5, 1.41e-4, 3.10e-4, 6.17e-4, 1.60e-3],
        ('b', 673): [5.28e-6, 1.13e-5, 2.36e-5, 5.37e-5],
        ('b', 723): [5.18e-6, 1.26e-5, 1.26e-5, 2.56e-5, 2.56e-5, 5.14e-5, 5.14e-5, 2.79e-4],
        ('b', 773): [5.40e-6, 1.21e-5, 2.56e-5, 2.56e-5, 5.37e-5, 5.37e-5, 1.30e-4, 2.67e-4, 2.84e-4, 6.08e-4]}

out = {}
for pn, (L, R, T, B) in panels.items():
    to_tau = lambda px: 10 ** (2 * (px - L) / (R - L))
    to_rate = lambda py: 10 ** (-2 - 5 * (py - T) / (B - T))
    legend_box = (L, L + 0.25 * (R - L), T, T + 0.22 * (B - T))
    for Tk, rgb in COL[pn].items():
        dist = np.sqrt(((a - np.array(rgb)) ** 2).sum(axis=2))
        m = dist < 70
        m[:, :L + 3] = False; m[:, R - 3:] = False; m[:T + 3] = False; m[B - 3:] = False
        m[int(legend_box[2]):int(legend_box[3]), int(legend_box[0]):int(legend_box[1])] = False
        m = ndi.binary_opening(m, structure=np.ones((5, 5)))
        lab, n = ndi.label(m)
        cents = ndi.center_of_mass(m, lab, range(1, n + 1))
        sizes = ndi.sum(m, lab, range(1, n + 1))
        pts = [(to_tau(cx), to_rate(cy), s) for (cy, cx), s in zip(cents, sizes) if s >= 20]
        expt = EXPT[(pn, Tk)]
        # match each tabulated rate to the nearest unused marker in log(rate)
        used, rows = set(), []
        for r in expt:
            best = None
            for j, (t, rr, s) in enumerate(pts):
                if j in used:
                    continue
                dlr = abs(np.log10(rr) - np.log10(r))
                if best is None or dlr < best[0]:
                    best = (dlr, j)
            if best is not None and best[0] < 0.15:
                used.add(best[1])
                rows.append(dict(expt_rate=r, tau_MPa=round(pts[best[1]][0], 2), dlog_rate=round(best[0], 3)))
            else:
                rows.append(dict(expt_rate=r, tau_MPa=None))
        out[f'{pn}_{Tk}'] = dict(d_um=7 if pn == 'a' else 3, T=Tk, n_markers=len(pts), rows=rows)
        print(pn, Tk, f'{len(pts)} markers;', [(r['expt_rate'], r['tau_MPa']) for r in rows])
Path(__file__).resolve().parents[1] / 'results' / 'mesoscale' / 'kap2024_figA1_digitized.json'.write_text(json.dumps(out, indent=1) + '\n')
