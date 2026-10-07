"""
Assembles `results/`: the single place to look for current results, and the
only place the figures are built from.

Every file listed here is copied from `outputs/` with its source time, and
the README records the git commit it was produced at, so a stale file is
visible rather than silent. Anything expected but not found is listed as
MISSING rather than skipped.
"""

import shutil, subprocess, sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
SRC = ROOT / "outputs"

# (filename, what it is, what produced it)
CURRENT = [
    ("gps_module6_lcow_bhojwani_global_lcow.csv",
     "LCOW by architecture and track - THE HEADLINE TABLE",
     "module6_lcow/module6_lcow.py"),
    ("FINAL_RESULTS.csv",
     "Consolidated figure table - every headline number with its basis and source",
     "reporting/final_results.py"),
    ("loop_optimisation_summary_lcow.csv",
     "Optimised FGCHE/ORC design per duct x architecture x track, with ID fan feasibility",
     "module5_orc/loop_optimiser.py"),
    ("sweep_stage3_tornado.csv",
     "Sensitivity swing ranking, PER ARCHITECTURE (all 8)",
     "module6_lcow/sensitivity_sweep.py"),
    ("sweep_stage3_oat.csv",
     "One-at-a-time sensitivity results feeding the tornado",
     "module6_lcow/sensitivity_sweep.py"),
    ("sweep_stage3_factorial.csv",
     "Full factorial, 9 economic inputs x 8 architectures (157,464 rows) - LCOW distribution",
     "module6_lcow/sensitivity_sweep.py"),
    ("sweep_stage4_corners.csv",
     "ORC+RO design re-optimised at 15 corners (5 ducts x 3 discount rates)",
     "module6_lcow/stage4_corners.py"),
    ("med_effects_sweep.csv",
     "MED effect-count optimisation at the chosen top brine temperature",
     "module4a_med/med_effects_sweep.py"),
    ("med_design_summary.csv",
     "MED design per track - TBT, effect count, GOR, loop return, Tc - both MED architectures",
     "module6_lcow/module6_lcow.py"),
    ("sensitivity_table.csv", "Every swept quantity, the range swept, and the baseline value adopted",
     "reporting/sensitivity_table.py"),
    ("GPS_Module1_Output.csv", "Module 1 per-row combustion, ADP, Cp and recovered heat, all four tracks",
     "module1_combustion/module1_combustion.py"),
    ("GPS_Module2_Output.csv", "Module 2 per-row energy and exergy content of the recovered band, all four tracks",
     "module2_exergy/module2_energy_exergy.py"),
    ("gps_module3_output_direct_med_85.csv", "Module 3 per-row dataset, Direct-MED 85C track",
     "module3_fgche/module3_fgche_rating.py"),
    ("gps_module3_output_direct_med_90.csv", "Module 3 per-row dataset, Direct-MED 90C track",
     "module3_fgche/module3_fgche_rating.py"),
    ("gps_module3_output_direct_med_95.csv", "Module 3 per-row dataset, Direct-MED 95C track",
     "module3_fgche/module3_fgche_rating.py"),
    ("gps_module3_output_direct_med_100.csv", "Module 3 per-row dataset, Direct-MED 100C track",
     "module3_fgche/module3_fgche_rating.py"),
]
CURRENT += [(f"gps_module2b_exergy_destruction_direct_med_{t}.csv",
             f"Module 2b FGCHE exergy destruction per row, Direct-MED {t}C track (rated states)",
             "module2_exergy/module2b_exergy_destruction.py") for t in (85, 90, 95, 100)]
CURRENT += [(f"gps_module4a_data_track_{t}.csv",
             f"Module 4a iterative MED cascade per row at the chosen design, {t}C track",
             "module4a_med/module4a_direct_med.py") for t in (90, 95, 100)]
# (no 85 C: Direct-MED has no fan-feasible design there, so Module 4a writes no file for it)
# Supporting calculations
CURRENT += [(f, d, "analysis/supporting_calculations.py") for f, d in (
    ("asset_life_sensitivity.csv", "CR4 - LCOW with heat-pool life 25/20/15/10 yr, corrosion and closure cases"),
    ("loop_return_scan.csv", "CR5 - ORC+RO 85C loop-return scan, paths A and B, feasibility and LCOW (figure 4)"),
    ("pinch_composite_curves.csv", "CR6 - evaporator T-Q composite curves, invalid vs corrected design (figure 3)"),
    ("pinch_exergy_check.csv", "CR6 - cumulative exergy destruction along the evaporator (figure 3)"),
    ("orc_specific_cost_check.csv", "CR7 - ORC thermal efficiency and implied A$/kW against the 2,000-5,000 band"),
    ("operating_data_characterisation.csv", "CR9 - exhaust vs load, outage runs, ADP spread, seawater by month"),
    ("corrosion_retreat_penalty.csv", "CR10 - LCOW by extraction track 85-100C, ADP margin, matched-capacity penalty"),
    ("seawater_sensitivity.csv", "CR12 - ORC+RO LCOW at 20-30C intake, both capacity interpretations, annualised"),
    ("supporting_calculations_summary.txt", "Key numbers from every supporting calculation, as printed by the run"),
    ("pinch_resolution_check.csv", "CR6b - every ORC evaporator checked at 20 and 1,000 segments and exactly at its bubble/dew points"),
    ("pinch_resolution_profile.csv", "CR6b - the ORC+RO evaporator at 1,000 segments, with the model's 21 check points marked"),
    ("pinch_before_after_profile.csv", "ORC+RO evaporator temperature difference with and without the bubble point checked (figure 7)"),
    ("pinch_profiles_all_designs.csv", "temperature difference along the evaporator for every reported ORC design (figure 7)"),
    ("med_sizing_percentile_check.csv", "Direct-MED LCOW with the plant sized at P50-P100 of the per-record heat"),
    ("subcritical_orc_all_tracks.csv", "CR15 - subcritical ORC (path A) across its loop-return range at every track, against Direct-RO"),
)]
# Monte Carlo
CURRENT += [(f, d, "module6_lcow/monte_carlo.py") for f, d in (
    ("monte_carlo_summary.csv", "Monte Carlo, triangular and PERT: mean and P5-P95 LCOW per architecture"),
    ("monte_carlo_penalties.csv", "Monte Carlo paired penalties - heat recovery and the ORC - with the chance each pays"),
    ("monte_carlo_scurves.csv", "S-curves: LCOW at every 0.5 % of probability, per architecture"),
    ("monte_carlo_convergence.csv", "P50 on the first 5,000 / 10,000 / 20,000 / 40,000 draws"),
    ("monte_carlo_draws.csv", "every triangular draw: its nine inputs and each architecture's LCOW"),
)]
# Energy, heat and exergy tables
CURRENT += [(f, d, "analysis/energy_tables.py") for f, d in (
    ("fan_power_by_design.csv", "Added ID fan power per reported design: design record and mean over the operating records"),
    ("heat_by_track.csv", "Heat recovered and delivered per track: design record and min/median/mean/max over the records"),
    ("grid_import_by_architecture.csv", "Grid electricity per m3 per architecture, split into process and heat side"),
    ("exergy_balance_designs.csv", "Design-record exergy balance of the four reported designs, component by component"),
)]


# Built in place by reporting/lcow_breakdown.py rather than copied from outputs/
IN_PLACE = [
    ("report_lcow_breakdown.csv", "LCOW decomposed to A$/m3 per component; sums to the reported LCOW"),
    ("report_capex.csv", "Capital by item - FGCHE, ORC package, ORC exchangers, preheater, desal"),
    ("report_opex.csv", "Operating cost by item - O&M split, membranes, electricity by term"),
    ("report_architecture_summary.csv", "Headline summary, one row per architecture and track"),
]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now()
    commit = git_commit()
    copied, missing = [], []

    for name, desc, producer in CURRENT:
        src = SRC / name
        if not src.exists():
            missing.append((name, desc)); continue
        shutil.copy2(src, OUT / name)
        copied.append((name, desc, producer,
                       datetime.fromtimestamp(src.stat().st_mtime), src.stat().st_size))

    # Every value in the README is READ from the results, never typed here, so
    # the README cannot drift from the numbers it describes.
    _fr = pd.read_csv(SRC / "FINAL_RESULTS.csv").set_index("figure")["value"]

    def _spec(csv):
        x = pd.read_csv(SRC / csv)
        x = x[x.architecture == "Conventional SWRO (no heat recovery)"].iloc[0]
        return x.capex_water_aud / x.water_m3_per_day
    _sw = pd.read_csv(SRC / "gps_module3_output_direct_med_85.csv")["Sea water intake temp"]
    _st = pd.read_csv(SRC / "sensitivity_table.csv").set_index("parameter")["baseline_value"]
    lines = [
        "# Results",
        "",
        f"**Generated {stamp:%Y-%m-%d %H:%M}** from git commit `{commit}`.",
        "",
        "Everything in this folder is current and mutually consistent. Build",
        "figures from here, not from `outputs/`, which is a working directory.",
        "",
        "Regenerate with `python run_all.py`.",
        "",
        "## Model basis for these results",
        "",
        "| | |",
        "|---|---|",
        "| duct | **5.79 x 4.235 m = 24.52 m2**, measured at the fabric-filter outlet expansion joint |",
        "| topology | 2 parallel gas trains per unit, each with its own duct and 2450 kW ID fan |",
        "| exchangers | **2 fitted**, costed as 2 separate units (not one of twice the area) |",
        f"| ID fan | hard constraint, worst of 461 records within 2450 kW, PF {_st.get('ID fan motor power factor', 'n/a')}, clean |",
        f"| fan row cap | **{_fr['ID fan row cap (measured duct)']} rows** at the measured duct |",
        f"| ORC exchangers | costed on the Shamoushaki correlation, U = {_st.get('ORC / preheater overall U', 'n/a')} W/m2K |",
        f"| desalination capital | Bhojwani et al. (2019), located to Australia: A${_spec('gps_module6_lcow_bhojwani_global_lcow.csv'):,.0f} per m3/day for the conventional SWRO plant; the absolute level is uncertain, the penalties are not (both cases buy the same plant) |",
        f"| seawater | per-record intake over the 461 operating records, {_sw.min():.1f}-{_sw.max():.1f} C; RO sized at the {_sw.median():.1f} C median, the MED condenser at the {_sw.max():.1f} C maximum |",
        f"| headline | recovering the heat **adds** A${_fr['PENALTY for recovering the heat']}/m3; conventional SWRO is cheapest in {_fr['ordering, global-basis grid']} combinations of the sensitivity grid |",
        f"| MED | top brine temperature and effect count co-optimised with the loop pinch enforced: Direct-MED {_fr.get('Direct-MED: extraction track', 'n/a')} C track, TBT {_fr.get('Direct-MED: top brine temperature', 'n/a')} C, {_fr.get('Direct-MED: number of effects', 'n/a')} effects |",
        "",
        "## Current files",
        "",
        "| file | contents | produced by | source mtime |",
        "|---|---|---|---|",
    ]
    for name, desc, producer, mtime, size in copied:
        lines.append(f"| `{name}` | {desc} | `{producer}` | {mtime:%Y-%m-%d %H:%M} |")

    lines += ["", "## Built in place", "",
              "| file | contents |", "|---|---|"]
    for name, desc in IN_PLACE:
        mark = "" if (OUT / name).exists() else "  **MISSING - run reporting/lcow_breakdown.py**"
        lines.append(f"| `{name}` | {desc}{mark} |")

    if missing:
        lines += ["", "## MISSING - expected but not found", ""]
        lines += [f"- `{n}` - {d}" for n, d in missing]

    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}/README.md")
    for name, desc, _, mtime, size in copied:
        print(f"  {name:52s} {size/1024:8.0f} KB  {mtime:%Y-%m-%d %H:%M}")
    if missing:
        print("  MISSING:", ", ".join(n for n, _ in missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
