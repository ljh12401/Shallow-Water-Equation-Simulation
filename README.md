# SWE Lake Model Course Project

This project implements the semester assignment Shallow Water Equation: a four-scenario lake model using the linearized shallow water equations and the supplied bathymetry grid.

## What Is Included

- `src/config.py`: physical constants, grid defaults, CFL-based time step.
- `src/scenarios.py`: the four wind/barrier scenarios.
- `src/model.py`: finite-difference shallow water solver for `zeta`, `U`, and `V`.
- `src/diagnostics.py`: velocity, vorticity, eddy kinetic energy, summary CSV, data export.
- `src/plots.py`: figures for the assignment questions.
- `tests/test_project.py`: data, scenario, stability, and output checks.

## Assumptions

- `bathymetry.txt` has shape `(40, 20)`.
- Depth `H=0` means land or closed boundary.
- Grid indices are Python zero-based indices. The required point `[25, 10]` is read as `zeta[:, 25, 10]`.
- `WX=10` represents the easterly wind used in the assignment text; `WY=10` represents northward wind.
- Because no `dx`, `dy`, or `dt` is provided in the lecture PDF, defaults are `dx=dy=1000 m` and `dt` is selected automatically from a CFL safety factor.

## Run Everything

```bash
python3 -m src.run_all
```

Optional overrides:

```bash
python3 -m src.run_all --steps 1000 --output-every 5 --dx 1000 --dy 1000
python3 -m src.run_all --dt 10
```

The run writes:

- `outputs/summary.csv`
- `outputs/data/scenario_*.npz`
- `outputs/figures/question_a_point_timeseries.png`
- `outputs/figures/question_c_hovmoller_transect_25.png`
- `outputs/figures/question_e_vorticity_eke.png`
- `outputs/figures/scenario_*_mean_std_maps.png`
- `outputs/figures/scenario_*_flow_snapshots.png`

## Assignment Question Mapping

- a) `question_a_point_timeseries.png` shows sea level `zeta` at `[25, 10]`.
- b) `scenario_*_mean_std_maps.png` shows mean and standard deviation of `zeta`, `u`, and `v`.
- c) `question_c_hovmoller_transect_25.png` compares `zeta` along transect `x=25`.
- d) `scenario_*_flow_snapshots.png` shows flow snapshots over time.
- e) `question_e_vorticity_eke.png` and `summary.csv` compare mean vorticity and eddy kinetic energy.

## Run Tests

```bash
python3 -m unittest discover -s tests
```

