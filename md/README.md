# MD calculations of Appendix A

All inputs use the Cu potential of Mishin et al. (2001), passed to LAMMPS as `-var pot`.
Settings are summarised in Section S7 of the Supporting Information. Commands below are
written for `mpirun`; on a cluster replace it with your launcher (e.g. `srun`).

```bash
POT=$(pwd)/potential/Cu_mishin1.eam.alloy
```

## 1. Generalised stacking-fault energy (`gsfe/`)

```bash
cd gsfe
bp=$(python3 -c "print(3.615/6**0.5)")
# partial path along [11-2]
mpirun -np 8 lmp -in in.gsfe.lammps -var pot $POT -var ox "1 1 -2" -var oy "-1 1 0" \
  -var nx 6 -var ny 10 -var pxexpr "v_a*sqrt(6)/2" -var pyexpr "v_a/sqrt(2)" \
  -var dir "[11-2]" -var dmax $(python3 -c "print(1.2*$bp)") -var nsteps 49 -var out gsfe_partial.dat
# full path along [-110]
mpirun -np 8 lmp -in in.gsfe.lammps -var pot $POT -var ox "-1 1 0" -var oy "-1 -1 2" \
  -var nx 10 -var ny 6 -var pxexpr "v_a/sqrt(2)" -var pyexpr "v_a*sqrt(6)/2" \
  -var dir "[-110]" -var dmax $(python3 -c "print(3.615/2**0.5)") -var nsteps 41 -var out gsfe_full.dat
```

## 2. Nanocrystalline polycrystals (`polycrystal/`)

Build 16-grain structures (needs Atomsk; set `ATOMSK` if it is not on the path). `d` is the
equivalent-sphere grain diameter in nm; `TAG=_s2` builds a second random seed.

```bash
cd polycrystal
./gen_poly.sh 3 4 5 6 8 10 12 14 16
TAG=_s2 ./gen_poly.sh 3 4 5 6 8 10
```

Tension to 10 % strain. `erate` is in 1/ps (5e-4 = 5e8 1/s); `nrun = 0.10/erate/0.002`
steps at a 2-fs time step; `ndump` sets the snapshot interval.

```bash
mkdir -p runs/d5_T300_r5e8
mpirun -np 32 lmp -in in.tension.lammps -var pot $POT -var data poly_d5.lmp -var seed 4928 \
  -var T 300 -var erate 0.0005 -var nrun 100000 -var ndump 20000 -var outdir runs/d5_T300_r5e8
```

Run names used by the analysis: `d{d}_T300_r5e8`, `d{d}s2_T300_r5e8` (second seed) and
`rate_d5_T{T}_r{erate}` for the rate-temperature series (T = 300-700 K;
erate = 0.0001, 0.0003, 0.001, 0.003). Then:

```bash
python ../../analysis/analyze_poly.py      # flow stress, DXA line content, HCP fraction
python ../../analysis/analyze_rate.py      # activation volume and energy
```

## 3. Quasi-static shear of bicrystals and NEB (`bicrystal/`)

Starting structures are in `structures/` (`gunzip -k`): three Sigma11 (557)/(77-1)
boundary structures (`c0_s30`, `c0_s03`, `cp_s21`; `cp_s12` relaxes to `c0_s30`) and
Sigma7 (415). `dstrain` is the shear-strain increment per step (negative for reverse shear).

```bash
cd bicrystal
mkdir -p runs/qs_set/c0_s30_reverse
mpirun -np 32 lmp -in in.quasistatic_strain.lammps -var pot $POT \
  -var data ../structures/Sigma11_c0_s30.data -var outdir runs/qs_set/c0_s30_reverse \
  -var dstrain -0.004 -var nsteps 80
python ../../analysis/analyze_qs_set.py    # first emission event per boundary
```

NEB for the emission event (step `k_c` = 21 for `c0_s30` reverse; initial states k = 17-20).
The emitted state is mapped into the cell of the initial state and relaxed, then a
climbing-image band is run with one replica per partition:

```bash
D=runs/neb_k20; mkdir -p $D
mpirun -np 8 lmp -in in.map_final.lammps -var pot $POT -var src runs/qs_set/c0_s30_reverse/qs.21.data \
  -var yz <tilt of step 20, column yz_A of qs.dat> -var outdata $D/final.data -var outdump $D/final.dump
awk 'NR>9' $D/final.dump > $D/tmp; (wc -l < $D/tmp; cat $D/tmp) > $D/final.coords
mpirun -np 64 lmp -partition 16x4 -in in.neb_emit.lammps -var pot $POT \
  -var initial runs/qs_set/c0_s30_reverse/qs.20.data -var finalcoords $D/final.coords -var outdir $D \
  -log $D/log.master -plog $D/log.replica
```

The NEB energy profiles used in the paper are in `../results/neb/` (`emit_k*`: 8 replicas;
`dense_k*`: 16 replicas).
