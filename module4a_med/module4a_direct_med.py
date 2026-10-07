"""
Module 4a: per-row MED cascade for Direct-MED - a VERIFICATION of the design
Module 6 chooses, not an independent model.

It reads each track's chosen top brine temperature (TBT) and effect count
from outputs/med_design_summary.csv (written by Module 6) and runs the full
iterative cascade - boiling-point elevation and temperature-dependent latent
heat, both per Wang et al. (Int. J. Heat Mass Transf. 54, 2011) - on every
operating record. That checks the closed-form GOR and mid-cascade latent
heat Module 6 designs with against the detailed cascade.

Rows whose seawater is warm enough that the chosen effect count no longer
fits the TBT-to-condenser range are reported as pinched out rather than
silently run at a sub-minimum per-effect drop.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import CoolProp.CoolProp as cp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.constants import TARGET_TEMPERATURES
from common.med import (MED_EVAP_EFFECTIVENESS, MED_DELTA_T_PER_EFFECT_C, condenser_temperature_c,
                        gor_for_effects, latent_heat_kj_kg, distillate_m3_per_day)

salinity_ppm = 35000.0          # typical seawater (literature); no site composition data
ncg_leak_per_effect = 0.0015    # non-condensable leak per shell, kg/s - UNCITED
vacuum_pump_efficiency = 0.65   # liquid-ring vacuum pump - UNCITED


def chosen_designs(out_dir):
    """Per-track (TBT, N) chosen by Module 6 for Direct-MED."""
    f = Path(out_dir) / "med_design_summary.csv"
    if not f.exists():
        raise FileNotFoundError("run Module 6 first - it writes med_design_summary.csv")
    d = pd.read_csv(f)
    d = d[(d["architecture"] == "Direct-MED") & (d["feasible"] == True)]
    return {float(r.track_c): (float(r.tbt_c), int(r.n_effects)) for r in d.itertuples()}


def cascade(q_in_kw, t_tbt, t_condenser, n_effects):
    """
    Iterative effect-to-effect cascade. Effect 1 receives q_in x e; each
    effect's vapour condenses in the next with the same transfer loss e.
    Returns distillate in kg/s, or None if the effects do not fit.
    """
    if int(np.floor((t_tbt - t_condenser) / MED_DELTA_T_PER_EFFECT_C)) < n_effects:
        return None
    t_step = (t_tbt - t_condenser) / n_effects
    q = q_in_kw * MED_EVAP_EFFECTIVENESS
    t_boil = t_tbt
    m_total = 0.0
    for _ in range(n_effects):
        b = (6.71 + 0.0634 * t_boil + 0.0000974 * t_boil ** 2) * 0.001
        c = (22.238 + 0.00959 * t_boil + 0.0000000942 * t_boil ** 2) * 1e-8
        bpe = salinity_ppm * (b + c * salinity_ppm) * 0.001
        l_v = latent_heat_kj_kg(t_boil - bpe)
        m = q / l_v
        m_total += m
        q = m * l_v * MED_EVAP_EFFECTIVENESS
        t_boil -= t_step
    return m_total


def run_module4a_desalination(module3_output_dir, architecture="direct_med"):
    designs = chosen_designs(module3_output_dir)
    for t_target in TARGET_TEMPERATURES:
        if float(t_target) not in designs:
            print(f"--- {t_target} C track: no feasible Direct-MED design - skipped ---\n")
            continue
        t_tbt, n_eff = designs[float(t_target)]
        input_csv = Path(module3_output_dir) / f"gps_module3_output_{architecture}_{t_target}.csv"
        df = pd.read_csv(input_csv)

        water_lh, water_closed_lh, vac_kw, pinched = [], [], [], 0
        for _, row in df.iterrows():
            q = row["heat_delivered_to_downstream_kw"]
            t_cond = condenser_temperature_c(row["Sea water intake temp"])
            m = cascade(q, t_tbt, t_cond, n_eff) if q > 0 else None
            if m is None:
                if q > 0:
                    pinched += 1
                water_lh.append(0.0); water_closed_lh.append(0.0); vac_kw.append(0.0)
                continue
            water_lh.append(m * 3600.0)
            water_closed_lh.append(distillate_m3_per_day(q, gor_for_effects(n_eff), t_tbt, t_cond) * 1000.0 / 24.0)
            p_cond = cp.PropsSI("P", "T", t_cond + 273.15, "Q", 0, "Water")
            vac_kw.append(n_eff * ncg_leak_per_effect * 0.28705 * (t_cond + 273.15)
                          * np.log(101325.0 / p_cond) / vacuum_pump_efficiency)

        out = df.copy()
        out["med_tbt_c"] = t_tbt
        out["med_effects_installed"] = n_eff
        out["water_produced_lh"] = water_lh
        out["water_closed_form_lh"] = water_closed_lh
        out["vacuum_pump_parasitic_kw"] = vac_kw
        out.to_csv(Path(module3_output_dir) / f"gps_module4a_data_track_{t_target}.csv", index=False)

        act = out[out["water_produced_lh"] > 0]
        print(f"--- {t_target} C track: TBT {t_tbt:.1f} C, {n_eff} effects ---")
        if len(act):
            ratio = act["water_closed_form_lh"].sum() / act["water_produced_lh"].sum()
            print(f"  mean water (iterative cascade): {act['water_produced_lh'].mean():,.0f} L/h")
            print(f"  closed-form / iterative water:  {ratio:.4f}  (Module 6 designs with the closed form)")
            print(f"  rows pinched out by warm seawater: {pinched} of {len(df)}")
            print(f"  mean vacuum draw: {act['vacuum_pump_parasitic_kw'].mean():.2f} kW\n")


if __name__ == "__main__":
    run_module4a_desalination(str(Path(__file__).resolve().parent.parent / "outputs"))
