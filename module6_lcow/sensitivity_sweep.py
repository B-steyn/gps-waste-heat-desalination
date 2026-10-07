"""
Sensitivity sweep (stage 3): the economic inputs varied around the
optimised designs.

Cheap, because the economic inputs do not require re-running the
thermodynamics: each design is fixed and only its costing changes. Runs a
one-at-a-time pass (for a tornado ranking) and a full factorial over nine
inputs at three values each (3^9 = 19,683 combinations), scoring every
architecture on the same combinations so that comparisons are paired.

Stage 4 (stage4_corners.py) re-optimises the DESIGN at the corners of this
space, to test whether the optimal configuration itself moves.

Only inputs are swept. Design choices (track, loop return, ORC path,
membrane case, MED effect count, MED sizing percentile) are optimised, not
swept: sweeping something the designer controls would imply it cannot be
chosen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import itertools
import numpy as np
import pandas as pd

import common.economic_constants as ec
from common.lcow import capital_recovery_factor, annual_operating_hours
from common.costing import desal_specific_capex_aud_per_m3_day, hx_purchased_cost_aud, med_capex_gor_factor

# Designs are READ FROM THE MODEL (the current Module 6 output), never typed
# in, so the sweep cannot drift from the model it tests; the baseline of every
# design is checked against Module 6's own LCOW below. Areas (not costs) are
# carried, because the Lang factor is itself a swept input and has to be
# re-applied per combination.
from common.constants import N_TRAINS_FITTED


def load_designs(output_dir, basis="bhojwani_global"):
    """Every architecture's best-track design, straight from Module 6."""
    df = pd.read_csv(Path(output_dir) / f"gps_module6_lcow_{basis}_lcow.csv")
    best = df.loc[df.groupby("architecture")["lcow_aud_per_m3"].idxmin()]
    out = []
    for _, r in best.iterrows():
        out.append({
            "architecture": r.architecture,
            "technology": r.technology,
            "track_c": r.track_c,
            "water_m3_per_day": r.water_m3_per_day,
            "sec_kwh_m3": r.sec_effective_kwh_m3,
            "orc_gross_kw": r.orc_gross_kw,
            "fan_kw": r.fan_kw,
            "loop_pump_kw": r.loop_pump_kw,
            # the heat side's annual-mean power, as Module 6 charges it
            "heat_side_kw": r.heat_side_kw,
            "hx_area_m2": r.hx_area_m2,
            "orc_evap_area_m2": (0.0 if pd.isna(r.get("orc_evap_area_m2", np.nan))
                                  else r.orc_evap_area_m2),
            "orc_cond_area_m2": (0.0 if pd.isna(r.get("orc_cond_area_m2", np.nan))
                                  else r.orc_cond_area_m2),
            "preheater_area_m2": (0.0 if pd.isna(r.get("preheater_area_m2", np.nan))
                                   else r.preheater_area_m2),
            "reported_lcow": r.lcow_aud_per_m3,
            "med_gor": (None if pd.isna(r.get("med_gor", np.nan)) else float(r.med_gor)),
            "capacity_factor": (1.0 if pd.isna(r.get("capacity_factor", np.nan)) else float(r.capacity_factor)),
        })
    return out

# --- sensitivities: (label, attribute, values, baseline) ---
# O&M rates: Ruiz-Garcia & Nuez (2017) bands. WACC: Gohdes et al. (2022),
# about 5 % (government-backed) to 13 % (merchant), nominal. Electricity:
# AEMO QED Q4-2025 low to the AER CY2024 Queensland average. Asset life: a
# one-directional downside (station closure can only shorten it).
def sweep_axes():
    return [
        ("O&M water rate", "oandm_water", [0.05, 0.06, 0.08], 0.06),
        ("O&M heat rate", "oandm_heat", [0.03, 0.04, 0.06], 0.04),
        ("Lang factor", "lang", [3.0, 3.5, 5.0], 3.5),
        ("Bhojwani cost error", "cost_error", [0.80, 1.00, 1.30], 1.00),
        ("WACC (nominal)", "wacc_nom", [0.05, 0.08, 0.13], 0.08),
        ("asset life (yr)", "life", [25.0, 15.0, 10.0], 25.0),
        ("membrane life (yr)", "membrane_life", [3.0, 5.0, 7.0], 5.0),
        ("electricity (A$/MWh)", "elec", [58.0, 65.0, 128.0], 65.0),
        ("desal availability", "desal_avail", [0.85, 0.95, 0.99], 0.95),
    ]


def evaluate(design, oandm_water, oandm_heat, lang, cost_error,
              wacc_nom, life, membrane_life, elec, desal_avail, anchor=ec.DESAL_COST_ANCHOR):
    """
    LCOW for one sensitivity combination, with the DESIGN held fixed.

    Deliberately re-implements the cost build-up inline rather than
    calling the Module 6 driver: the driver reads module-level constants,
    and a sweep must vary them per-combination without mutating global
    state (which would make results order-dependent and unreproducible).
    """
    # The operating-hours rule follows the TECHNOLOGY: MED is heat-capped at
    # the station's availability (6,675 h); RO runs through boiler outages on
    # grid power (8,322 h). The drift guard against Module 6 (below) checks
    # this and every other term at the baseline.
    rule = "DIRECT_MED" if design["technology"].upper() == "MED" else "ORC_RO"
    hours = annual_operating_hours(rule, desal_own_availability=desal_avail)
    # MED runs part-loaded - the P90 capacity factor; 1.0 for RO
    water_yr = design["water_m3_per_day"] * design.get("capacity_factor", 1.0) * hours["total_h"] / 24.0

    # CAPEX - every exchanger priced from AREA so the swept Lang factor
    # applies to all of them, not just the FGCHE. The FGCHE is n separate
    # units (one per gas train), which Shamoushaki's non-linearity makes
    # materially different from one unit of n times the area.
    def installed_from_area(area_m2, n_units=1):
        if area_m2 <= 0:
            return 0.0
        pec_ = n_units * hx_purchased_cost_aud(area_m2 / n_units, "316L") * cost_error
        return pec_ * lang * ec.LOCATION_FACTOR_AU * (1 + ec.CONTINGENCY + ec.EPC_INDIRECT)

    hx_installed = installed_from_area(design["hx_area_m2"], N_TRAINS_FITTED)
    orc_hx = (installed_from_area(design["orc_evap_area_m2"])
              + installed_from_area(design["orc_cond_area_m2"]))
    preheater = installed_from_area(design["preheater_area_m2"])
    orc = design["orc_gross_kw"] * ec.ORC_SPECIFIC_COST_BASELINE
    heat_capex = hx_installed + orc + orc_hx + preheater
    water_capex = (design["water_m3_per_day"]
                   * desal_specific_capex_aud_per_m3_day(design["water_m3_per_day"],
                                                          design["technology"], anchor)
                   * cost_error
                   # MED's performance-ratio correction for the GOR run
                   * (med_capex_gor_factor(design["med_gor"]) if design.get("med_gor") else 1.0))

    # annualise on the REAL rate
    r_real = (1 + wacc_nom) / (1 + ec.INFLATION) - 1
    crf = capital_recovery_factor(r_real, life)
    annual_capital = (heat_capex + water_capex) * crf

    # OPEX - membranes are a share of the water O&M rate, scaled by life
    life_factor = ec.MEMBRANE_LIFE_YEARS / membrane_life
    mem_share = ec.MEMBRANE_SHARE_OF_RO_OPEX * life_factor
    water_om = water_capex * oandm_water * ((1 - ec.MEMBRANE_SHARE_OF_RO_OPEX) + mem_share)
    heat_om = heat_capex * oandm_heat

    process_kw = design["sec_kwh_m3"] * design["water_m3_per_day"] * design.get("capacity_factor", 1.0) / 24.0
    # the heat side runs only while there is flue gas (preheated hours); RO's
    # process demand runs all its hours, as in Module 6. It is Module 6's
    # annual mean over the operating records: no swept input touches the fan,
    # the pump or the ORC cycle, so it is carried rather than rebuilt.
    heat_kw = design["heat_side_kw"]
    elec_cost = -(heat_kw * hours["preheated_h"] - process_kw * hours["total_h"]) / 1000.0 * elec

    annual = annual_capital + water_om + heat_om + elec_cost
    return annual / water_yr


def stage3_oat(design):
    """One-at-a-time: each axis varied alone, everything else at baseline."""
    axes = sweep_axes()
    base_kwargs = {a[1]: a[3] for a in axes}
    base = evaluate(design, **base_kwargs)

    rows = []
    for label, attr, values, default in axes:
        for v in values:
            kw = dict(base_kwargs)
            kw[attr] = v
            lcow = evaluate(design, **kw)
            rows.append({"parameter": label, "value": v, "lcow": lcow,
                          "delta_vs_base": lcow - base,
                          "pct_vs_base": (lcow / base - 1) * 100.0,
                          "is_baseline": v == default})
    return base, pd.DataFrame(rows)


def stage3_factorial(design, max_combinations=200000):
    """Full factorial over every axis - cheap, no thermodynamics involved."""
    axes = sweep_axes()
    names = [a[1] for a in axes]
    grids = [a[2] for a in axes]
    total = int(np.prod([len(g) for g in grids]))
    if total > max_combinations:
        raise ValueError(f"{total} combinations exceeds the {max_combinations} guard")
    out = []
    for combo in itertools.product(*grids):
        kw = dict(zip(names, combo))
        out.append({**kw, "lcow": evaluate(design, **kw)})
    return pd.DataFrame(out), total


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    out = root / "outputs"
    designs = load_designs(out)

    all_oat, all_swing, all_fac = [], [], []
    for d in designs:
        base, oat = stage3_oat(d)
        # the sweep's own baseline must reproduce Module 6's reported LCOW,
        # or the two have drifted again
        drift = abs(base - d["reported_lcow"]) / d["reported_lcow"] * 100
        flag = "" if drift < 2.0 else f"   <-- DRIFT {drift:.1f}% vs Module 6"
        print(f"{d['architecture']:34s} track {str(d['track_c']):>5s}  "
              f"baseline A${base:6.3f}/m3 (Module 6: A${d['reported_lcow']:6.3f}){flag}")

        oat.insert(0, "architecture", d["architecture"])
        all_oat.append(oat)
        sw = (oat.groupby("parameter")
                 .agg(low=("lcow", "min"), high=("lcow", "max"))
                 .assign(swing=lambda x: x.high - x.low,
                         pct_of_base=lambda x: (x.high - x.low) / base * 100.0,
                         baseline=base, architecture=d["architecture"])
                 .sort_values("swing", ascending=False).reset_index())
        all_swing.append(sw)

        fac, total = stage3_factorial(d)
        fac.insert(0, "architecture", d["architecture"])
        all_fac.append(fac)

    oat_df = pd.concat(all_oat, ignore_index=True)
    swing_df = pd.concat(all_swing, ignore_index=True)
    fac_df = pd.concat(all_fac, ignore_index=True)

    oat_df.to_csv(out / "sweep_stage3_oat.csv", index=False)
    fac_df.to_csv(out / "sweep_stage3_factorial.csv", index=False)
    swing_df.to_csv(out / "sweep_stage3_tornado.csv", index=False)
    print(f"\n{len(fac_df):,} factorial rows across {len(designs)} architectures")

    print("\n" + "=" * 78)
    print("TORNADO - recommended architecture")
    print("=" * 78)
    rec = swing_df[swing_df.architecture == designs[0]["architecture"]]
    for a in swing_df.architecture.unique():
        if "Conventional SWRO" in a or "Direct-RO" in a or "orc_ro_current" in a:
            t = swing_df[swing_df.architecture == a]
            print(f"\n{a}  (baseline A${t.baseline.iloc[0]:.3f}/m3)")
            for r in t.head(5).itertuples():
                print(f"   {r.parameter:24s} A${r.low:6.3f} - A${r.high:6.3f}  "
                      f"swing A${r.swing:5.3f}  ({r.pct_of_base:5.1f}% of base)")

    print("\n" + "=" * 78)
    print("LCOW DISTRIBUTION across the full grid, by architecture")
    print("=" * 78)
    for a in fac_df.architecture.unique():
        q = fac_df[fac_df.architecture == a]["lcow"].quantile([0.05, 0.5, 0.95])
        print(f"  {a:34s} P5 A${q.iloc[0]:6.3f}   P50 A${q.iloc[1]:6.3f}   P95 A${q.iloc[2]:6.3f}")
