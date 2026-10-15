#!/bin/bash
# Build 16-grain random Voronoi Cu polycrystals with atomsk.
# d = equivalent-sphere grain diameter (nm); box side L = d / (6/(16*pi))^(1/3) = 2.031 d.
set -euo pipefail
cd "$(dirname "$0")"
AT=${ATOMSK:-atomsk}
[ -f cu_unit.xsf ] || $AT --create fcc 3.615 Cu cu_unit.xsf -v 0
for d in "$@"; do
  L=$(python3 -c "print(round($d*10/(6/(16*3.141592653589793))**(1/3),3))")   # Angstrom
  printf "box %s %s %s\nrandom 16\n" $L $L $L > seed_d${d}${TAG:-}.txt
  rm -f poly_d${d}${TAG:-}.lmp
  $AT --polycrystal cu_unit.xsf seed_d${d}${TAG:-}.txt poly_d${d}${TAG:-}.lmp -wrap -v 0
  echo "d=$d nm L=$L A atoms=$(awk '/atoms/{print $1; exit}' poly_d${d}${TAG:-}.lmp)"
done
