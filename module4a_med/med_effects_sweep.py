"""
LCOW against MED effect count at the chosen top brine temperature - a
DIAGNOSTIC of the design Module 6 selects, not a separate optimisation.

Module 6 co-optimises the top brine temperature (TBT) and the number of
effects N. This script reports how LCOW varies with N at the TBT Module 6
chose, through Module 6's own direct_med_design - one cost implementation,
not two - and checks that the lowest-LCOW N is the one Module 6 chose.

The MED-only column (med_capital_and_om_aud_per_m3) is reported alongside,
but the design is chosen on FULL LCOW. The two can disagree: the heat-side
cost is fixed at a given TBT, so extra effects dilute it and full LCOW
favours more effects than MED capital alone would.
"""

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from common.lcow import capital_recovery_factor
from common.economic_constants import WACC_REAL, ASSET_LIFE_WATER_YEARS, OANDM_RATE_MED_RO
from common.med import max_feasible_effects
from common.costing import med_capex_gor_factor


def main():
    sp = importlib.util.spec_from_file_location(
        "m6", ROOT / "module6_lcow" / "module6_lcow.py")
    m6 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m6)
    out, data = ROOT / "outputs", ROOT / "data"

    d = pd.read_csv(out / "med_design_summary.csv")
    dm = d[(d["architecture"] == "Direct-MED") & (d["feasible"] == True)]
    if dm.empty:
        print("no feasible Direct-MED design - nothing to sweep")
        return
    ch = dm.loc[dm["lcow_aud_per_m3"].idxmin()]
    track, tbt, t_cond = float(ch.track_c), float(ch.tbt_c), float(ch.t_condenser_c)
    n_max = max_feasible_effects(tbt, t_cond)
    crf = capital_recovery_factor(WACC_REAL, ASSET_LIFE_WATER_YEARS)

    rows = []
    for n in range(1, 17):
        if n > n_max:
            rows.append({"n_effects": n, "feasible": False, "tbt_c": tbt})
            continue
        r = m6.direct_med_design(track, str(data), str(out), force_tbt=tbt, force_n=n)
        if not r["feasible"]:
            rows.append({"n_effects": n, "feasible": False, "tbt_c": tbt})
            continue
        rec = r["record"]
        cw, wy = rec["capex_water_aud"], rec["water_m3_per_year"]
        rows.append({"n_effects": n, "feasible": True, "tbt_c": tbt, "gor": r["gor"],
                     "water_m3_per_day": rec["water_m3_per_day"], "water_m3_per_year": wy,
                     "capex_aud": cw, "specific_capex_aud_per_m3_day": cw / rec["water_m3_per_day"],
                     "capex_gor_factor": med_capex_gor_factor(r["gor"]),
                     "med_capital_and_om_aud_per_m3": (cw * crf + cw * OANDM_RATE_MED_RO) / wy,
                     "lcow_aud_per_m3": rec["lcow_aud_per_m3"]})

    df = pd.DataFrame(rows)
    df.to_csv(out / "med_effects_sweep.csv", index=False)
    ok = df[df["feasible"] == True]
    print(f"Direct-MED, {track:.0f} C track, TBT {tbt:.1f} C, condenser {t_cond:.1f} C -> "
          f"at most {n_max} effects\n")
    print(f"{'N':>3} {'GOR':>6} {'m3/day':>8} {'GOR capex factor':>17} {'MED cap+O&M':>12} {'LCOW':>8}")
    for r in ok.itertuples():
        print(f"{r.n_effects:>3} {r.gor:>6.2f} {r.water_m3_per_day:>8.0f} {r.capex_gor_factor:>17.3f} "
              f"{r.med_capital_and_om_aud_per_m3:>12.3f} {r.lcow_aud_per_m3:>8.3f}")
    best = ok.loc[ok["lcow_aud_per_m3"].idxmin()]
    print(f"\nlowest LCOW at N = {best.n_effects:.0f} (GOR {best.gor:.2f}); "
          f"Module 6 chose N = {int(ch.n_effects)} - "
          f"{'consistent' if int(best.n_effects) == int(ch.n_effects) else 'MISMATCH'}")
    print(f"wrote {out / 'med_effects_sweep.csv'}")


if __name__ == "__main__":
    main()
