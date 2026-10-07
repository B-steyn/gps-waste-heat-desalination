"""
Runs the whole model in dependency order, so every output is regenerated
from the current code in one pass.

  1  Module 1           per-record combustion, acid dew point, gas heat capacity
  2  Module 2           energy and exergy content of the recovered heat band
  3  Optimiser          exchanger and ORC designs, duct x architecture x track  [slow]
  4  MED design loop    Module 6 chooses the MED designs; Module 3 re-rates the
                        Direct-MED exchanger at the chosen loop returns; repeat
                        until the choice stops changing (the P90 sizing ratio
                        comes from Module 3, which depends on the loop return)
  5  Module 2b          exchanger exergy destruction from Module 3's rated states
  6  Module 4a          iterative MED cascade at the chosen designs (a check)
  7  LCOW breakdown     LCOW decomposed by component, capital and operating cost
  8  MED effects        LCOW against effect count at the chosen TBT
  9  Stage 3 sweep      tornado and full factorial, all architectures
 10  Monte Carlo        triangular LCOW distribution, with a PERT check
 11  Stage 4 corners    the ORC+RO design re-optimised at the corners       [slow]
 12  Supporting calcs   loop-return scans, asset life, track retreat, seawater,
                        operating data, pinch resolution, subcritical ORC scan
 13  Energy tables      fan power, heat by track, grid import, exergy balances
 14  Final results      FINAL_RESULTS.csv: every headline figure, computed
 15  Sensitivity table  every swept input, its range and its baseline
 16  Collect results    the report-facing outputs copied into results/
 17  Figures            results/figures, drawn from results/ only
 18  Independent checks validation/independent_checks.py; stops the run on a FAIL

    python run_all.py                 full run (about 15 minutes on 8 cores)
    python run_all.py --skip-optimiser  reuse the optimiser output
    python run_all.py --skip-slow     reuse the optimiser and Stage 4 outputs
"""

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "outputs"
DATA = ROOT / "data"


def step(label, script):
    t = time.time()
    print(f"\n=== {label} ===", flush=True)
    r = subprocess.run([PY, str(ROOT / script)], cwd=ROOT)
    if r.returncode:
        sys.exit(f"FAILED at: {label} (exit {r.returncode})")
    print(f"    [{label}] {time.time() - t:.0f}s", flush=True)


def _load(rel, name):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def med_design_loop(max_iter=4):
    """Choose MED designs, re-rate Module 3 at them, repeat until stable."""
    from common.constants import ORC_EVAPORATOR_PINCH_C
    m6 = _load("module6_lcow/module6_lcow.py", "m6")
    m3 = _load("module3_fgche/module3_fgche_rating.py", "m3")
    m3.active_architecture = "direct_med"

    def choice():
        # NaN-safe: infeasible tracks carry NaN TBT and N, and NaN never equals
        # NaN, so they are filled before the designs are compared.
        d = pd.read_csv(OUT / "med_design_summary.csv")
        key = d.reindex(columns=["architecture", "track_c", "tbt_c", "n_effects"]).fillna(-1.0)
        return d, tuple(map(tuple, key.values.tolist()))

    m6.run(str(OUT), anchor="bhojwani_global")
    d, prev = choice()
    for i in range(max_iter):
        dm = d[d["architecture"] == "Direct-MED"]
        lr = {float(r.track_c): (float(r.loop_return_c) if r.feasible else float(r.tbt_floor_c) + ORC_EVAPORATOR_PINCH_C)
              for r in dm.itertuples()}
        print(f"  iteration {i + 1}: re-rating Module 3 at loop returns {lr}", flush=True)
        m3.execute_master_module3_pipeline(str(DATA / "GPS_Operating_Data_Clean.csv"),
                                            str(DATA / "GPS_Water_related_data.csv"), str(OUT),
                                            loop_return_c_override=lr)
        m6.run(str(OUT), anchor="bhojwani_global")
        d, now = choice()
        if now == prev:
            print(f"  MED design converged after {i + 1} re-rating(s)", flush=True)
            break
        prev = now
    else:
        print("  WARNING: MED design did not converge - check med_design_summary.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true",
                    help="reuse the existing optimiser AND Stage 4 outputs")
    ap.add_argument("--skip-optimiser", action="store_true",
                    help="reuse the existing optimiser output only")
    args = ap.parse_args()
    t0 = time.time()
    step("Module 1", "module1_combustion/module1_combustion.py")
    step("Module 2", "module2_exergy/module2_energy_exergy.py")
    if not (args.skip_slow or args.skip_optimiser):
        step("Optimiser", "module5_orc/loop_optimiser.py")
    print("\n=== MED design loop + Module 6 ===", flush=True)
    med_design_loop()
    step("Module 2b", "module2_exergy/module2b_exergy_destruction.py")
    step("Module 4a", "module4a_med/module4a_direct_med.py")
    step("LCOW breakdown", "reporting/lcow_breakdown.py")
    step("MED effects", "module4a_med/med_effects_sweep.py")
    step("Stage 3 sweep", "module6_lcow/sensitivity_sweep.py")
    step("Monte Carlo", "module6_lcow/monte_carlo.py")
    if not args.skip_slow:
        step("Stage 4", "module6_lcow/stage4_corners.py")
    step("Supporting calculations", "analysis/supporting_calculations.py")
    step("Energy tables", "analysis/energy_tables.py")
    step("Final results", "reporting/final_results.py")
    step("Sensitivity table", "reporting/sensitivity_table.py")
    step("Collect results", "reporting/collect_results.py")
    step("Figures", "reporting/figures.py")
    step("Independent checks", "validation/independent_checks.py")
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
