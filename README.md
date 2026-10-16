# Gladstone waste-heat desalination model

A thermo-economic model of recovering low-grade heat from the flue gas of
Gladstone Power Station (Queensland) and using it to produce desalinated
water. It is the code behind Brendan Steyn's undergraduate thesis
(EGH490, Queensland University of Technology, 2026).

The heat is taken from the flue-gas duct just downstream of each fabric
filter, through a flue-gas exchanger (FGCHE) and a closed water loop. The
model compares desalination architectures on their levelised cost of water
(LCOW, A$/m³):

| architecture | what the recovered heat does |
|---|---|
| Direct-MED | drives a multi-effect distillation plant directly |
| ORC+MED | drives an organic Rankine cycle (ORC); its condenser heat drives MED |
| ORC+RO (current, advanced membranes) | drives an ORC whose power and condenser heat feed reverse osmosis (RO); 38 °C and 50 °C membrane limits |
| Direct-RO | preheats the RO feed, with no ORC (comparison case) |
| Conventional SWRO | nothing: grid-powered seawater RO at the same capacity (comparison case) |

Direct-RO and conventional SWRO split the cost of ORC+RO into the cost of
recovering the heat and the cost of the ORC.

## Running it

Python 3.13, with the packages in `requirements.txt`:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then, from the repository root:

```
python module0_data/clean_data.py   # once: raw_data/*.xlsx -> data/*.csv
python run_all.py                   # the whole model, about 15 minutes on 8 cores
```

`run_all.py --skip-optimiser` reuses the optimiser's output, and
`--skip-slow` also reuses Stage 4's. The pinch-check tests run separately:
`python validation/pinch_check_tests.py`.

## Data

The station's operating data are not included. They were supplied by the
station for this thesis and are not published with the code. To run the
model, place the two source spreadsheets in `raw_data/`:

- `GPS_Data_received_v2.xlsx`, sheet `Flue Gas`: six-hourly unit load, and
  per side (A and B) the flue-gas O₂, fabric-filter outlet temperature and
  pressure, dust burden and ID fan current, with the total gas flow;
- `GPS_Water_related_data.xlsx`: seawater intake temperature on the same
  six-hourly timestamps.

Module 0 drops the idle and shutdown records (unit load below 1 MW) and
writes `data/GPS_Operating_Data_Clean.csv` and
`data/GPS_Water_related_data.csv`, which every other module reads.

## Layout

The pipeline runs in this order (`run_all.py`):

| step | file | what it does |
|---|---|---|
| 0 | `module0_data/clean_data.py` | cleans the station spreadsheets (run once, before `run_all.py`) |
| 1 | `module1_combustion/module1_combustion.py` | per-record combustion, acid dew point, gas heat capacity, heat recovered at each track |
| 2 | `module2_exergy/module2_energy_exergy.py` | energy and exergy content of the recovered heat |
| 3 | `module5_orc/loop_optimiser.py` | exchanger and ORC designs for every duct, architecture and track (slow) |
| 4 | `module6_lcow/module6_lcow.py` with `module3_fgche/module3_fgche_rating.py` | LCOW for every architecture and track; the MED designs and the Direct-MED exchanger are iterated to a fixed point |
| 5 | `module2_exergy/module2b_exergy_destruction.py` | exergy destroyed in the exchanger and loop |
| 6 | `module4a_med/module4a_direct_med.py` | the MED cascade, record by record, as a check on the chosen designs |
| 7 | `reporting/lcow_breakdown.py` | LCOW, capital and operating cost broken down by item |
| 8 | `module4a_med/med_effects_sweep.py` | LCOW against the number of MED effects |
| 9 | `module6_lcow/sensitivity_sweep.py` | one-at-a-time (tornado) and full-factorial sensitivity |
| 10 | `module6_lcow/monte_carlo.py` | Monte Carlo LCOW distributions and paired penalties |
| 11 | `module6_lcow/stage4_corners.py` | the ORC+RO design re-optimised at the corners of the sensitivity space (slow) |
| 12 | `analysis/supporting_calculations.py` | supporting calculations CR4–CR15 (below) |
| 13 | `analysis/energy_tables.py` | fan power, heat by track, grid import and exergy balances |
| 14 | `reporting/final_results.py` | `FINAL_RESULTS.csv`: every headline figure, with its basis and source |
| 15 | `reporting/sensitivity_table.py` | every swept input, its range and its baseline |
| 16 | `reporting/collect_results.py` | copies the report-facing outputs into `results/` |
| 17 | `reporting/figures.py` | the figures, drawn from `results/` only |
| 18 | `validation/independent_checks.py` | independent re-calculation of the model's chain; stops the run on a failure |

The shared physics and economics are in `common/`:

| file | contents |
|---|---|
| `constants.py`, `economic_constants.py` | every physical, site and economic input, in one place |
| `coal_combustion.py`, `gas_properties.py`, `fluid_properties.py` | combustion, acid dew point, flue-gas and water properties |
| `hx_loop_physics.py`, `ntu_rating.py`, `fgche_sizing.py` | exchanger geometry, heat transfer, sizing and rating |
| `fan_margin.py` | the existing ID fans' spare power as a cap on exchanger depth, and the record-mean fan and heat factors |
| `loop_optimisation.py` | choice of the loop return temperature |
| `orc_cycle.py`, `pinch_check.py` | the ORC cycle and its evaporator pinch check |
| `med.py`, `ro_thermal.py` | the MED cascade and RO specific energy with feed preheat |
| `costing.py`, `lcow.py` | equipment costing and the levelised cost of water |
| `exergy_analysis.py` | component exergy destruction |

The three ORC paths are in `module5_orc/`: A subcritical, B transcritical,
and C a trilateral flash cycle (`path_c_tfc.py`: cycle-level functions only,
not run by `run_all.py` and not costed).

## Conventions

- **Costs** are constant 2026 Australian dollars, discounted at a real rate.
- **Extraction tracks** are the exchanger's gas outlet temperature: 100, 95,
  90 and 85 °C.
- **Design record.** Equipment is sized and priced at the operating record
  with the largest flue-gas flow. The ID fan limit is checked against every
  record, and annual energy (ORC output, added fan power) is the mean over
  the operating records (`module6_lcow.heat_side_annual_kw`).
- **Desalination capital** is costed on one basis for both MED and SWRO:
  Bhojwani et al. (2019), escalated, converted to AUD and moved to Australia
  with a location factor (`common/costing.py`). The `anchor` arguments in
  some function signatures accept only this basis.
- Inputs without a literature source are marked `UNCITED` where they are
  defined, and the swept ones appear in `results/sensitivity_table.csv`.

## Outputs

`outputs/` holds every intermediate and final file. `results/` holds the
report-facing subset, with a README listing each file, what produced it and
when, and `results/figures/`. `results/FINAL_RESULTS.csv` holds every
headline figure, computed from the outputs rather than typed. Neither
folder is committed; `run_all.py` regenerates both.

## Validation

- `validation/independent_checks.py` (the last step of `run_all.py`)
  recomputes the defining calculations from first principles, without the
  model's own functions, and compares them with what the model reported.
  Each check is PASS, FLAG (right for its assumption, but the assumption is
  questionable; the size is stated) or FAIL. A full run gives 46 PASS,
  6 FLAG and 0 FAIL.
- `validation/pinch_check_tests.py` tests that the evaporator pinch check
  catches the cases it should, against designs frozen in
  `validation/fixtures/`.

