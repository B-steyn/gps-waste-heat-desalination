"""
The LCOW decomposed, so every number can be traced to the physics or cost
basis that produced it. Four tables, written to results/:

  report_architecture_summary.csv  headline - one row per architecture x track
  report_lcow_breakdown.csv        LCOW decomposed to A$/m3 per component
  report_capex.csv                 capital by item
  report_opex.csv                  operating cost by item

The breakdown is the load-bearing one. LCOW is an aggregate, and the
central result is about WHERE the cost sits: the cost of recovering the
heat is almost all the capital and O&M of the heat-recovery plant, against
a preheat saving of under a cent per m3. That is only visible decomposed,
so the decomposition sums exactly back to the reported LCOW and is
asserted to do so.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import common.economic_constants as ec
from common.costing import hx_purchased_cost_aud, installed_cost
from common.lcow import capital_recovery_factor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"


def decompose(row, crf):
    """
    LCOW split into A$/m3 by component. Sums exactly to lcow_aud_per_m3.

    Sign convention: costs positive, credits negative (the ORC's output is
    the only credit). The heat side's power is its annual mean over the
    operating records, as Module 6 charges it.
    """
    wy = row.water_m3_per_year
    hours = row.operating_hours
    heat_hours = row.preheated_hours    # the heat side runs only with flue gas
    price = ec.ELECTRICITY_AUD_PER_MWH / 1000.0        # A$/kWh

    pre_capex = 0.0
    if not pd.isna(row.get("preheater_area_m2", np.nan)):
        pre_capex = installed_cost(hx_purchased_cost_aud(row.preheater_area_m2))
    orc_pkg = row.capex_orc_package_aud or 0.0
    orc_hx = row.capex_orc_exchangers_aud or 0.0
    fgche = row.capex_heat_recovery_aud - orc_pkg - orc_hx - pre_capex
    water_capex = row.capex_water_aud

    heat_capex = row.capex_heat_recovery_aud
    life_factor = 1.0
    mem_share = ec.MEMBRANE_SHARE_OF_RO_OPEX * life_factor
    is_ro = row.technology.upper() != "MED"

    items = [
        ("CAPEX", "FGCHE (flue-gas exchanger)", fgche * crf / wy),
        ("CAPEX", "ORC package (turbine/pump/gen)", orc_pkg * crf / wy),
        ("CAPEX", "ORC evaporator + condenser", orc_hx * crf / wy),
        ("CAPEX", "loop-seawater preheater", pre_capex * crf / wy),
        ("CAPEX", "desalination plant", water_capex * crf / wy),
        ("OPEX", "O&M heat-recovery pool", heat_capex * ec.OANDM_RATE_ORC_HX / wy),
        ("OPEX", "O&M desalination (ex-membranes)",
         water_capex * ec.OANDM_RATE_MED_RO * (1 - (mem_share if is_ro else 0.0)) / wy),
        ("OPEX", "membrane replacement",
         (water_capex * ec.OANDM_RATE_MED_RO * mem_share / wy) if is_ro else 0.0),
        # heat-side power at its annual mean over the operating records
        ("ELECTRICITY", "ID fan parasitic", row.fan_electric_kw * heat_hours * price / wy),
        ("ELECTRICITY", "loop circulating pump", row.loop_pump_kw * heat_hours * price / wy),
        ("ELECTRICITY", "desalination process demand", row.process_kw * hours * price / wy),
        ("ELECTRICITY", "ORC generation credit", -row.orc_mean_kw * heat_hours * price / wy),
    ]
    return items


def main():
    OUT.mkdir(exist_ok=True)
    crf = capital_recovery_factor(ec.WACC_REAL, ec.ASSET_LIFE_WATER_YEARS)
    frames = {}

    for basis in (ec.DESAL_COST_ANCHOR,):
        src = ROOT / "outputs" / f"gps_module6_lcow_{basis}_lcow.csv"
        if not src.exists():
            continue
        df = pd.read_csv(src)
        frames[basis] = df

    rows_break, rows_capex, rows_opex = [], [], []
    for basis, df in frames.items():
        for _, r in df.iterrows():
            items = decompose(r, crf)
            total = sum(v for _, _, v in items)
            # the decomposition must reconstruct the reported LCOW exactly
            # (with equal pool lives it is exact, so any gap is an error)
            assert abs(total - r.lcow_aud_per_m3) < 1e-6, (
                f"{basis} {r.architecture}: decomposition {total:.4f} != "
                f"reported {r.lcow_aud_per_m3:.4f}")
            # The LCOW total is carried as a COLUMN on every row, never as a
            # pseudo-component row, so `aud_per_m3` holds components only and
            # is always safe to sum.
            for grp, name, val in items:
                rows_break.append({
                    "cost_basis": basis, "architecture": r.architecture,
                    "track_c": r.track_c, "group": grp, "component": name,
                    "aud_per_m3": round(val, 4),
                    "share_of_lcow_pct": round(100 * val / r.lcow_aud_per_m3, 2),
                    "lcow_total_aud_per_m3": round(r.lcow_aud_per_m3, 4),
                })

            pre_capex = (installed_cost(hx_purchased_cost_aud(r.preheater_area_m2))
                         if not pd.isna(r.get("preheater_area_m2", np.nan)) else 0.0)
            rows_capex.append({
                "cost_basis": basis, "architecture": r.architecture, "track_c": r.track_c,
                "fgche_aud": round(r.capex_heat_recovery_aud - (r.capex_orc_package_aud or 0)
                                    - (r.capex_orc_exchangers_aud or 0) - pre_capex),
                "orc_package_aud": round(r.capex_orc_package_aud or 0),
                "orc_exchangers_aud": round(r.capex_orc_exchangers_aud or 0),
                "preheater_aud": round(pre_capex),
                "desalination_aud": round(r.capex_water_aud),
                "total_capex_aud": round(r.total_capex_aud),
                "annual_capital_charge_aud": round(r.annual_capital_charge_aud),
                "hx_area_m2": round(r.hx_area_m2, 1),
                "orc_evap_area_m2": (round(r.orc_evap_area_m2, 1)
                                      if not pd.isna(r.get("orc_evap_area_m2", np.nan)) else None),
                "orc_cond_area_m2": (round(r.orc_cond_area_m2, 1)
                                      if not pd.isna(r.get("orc_cond_area_m2", np.nan)) else None),
                "preheater_area_m2": (round(r.preheater_area_m2, 1)
                                       if not pd.isna(r.get("preheater_area_m2", np.nan)) else None),
            })

            hours, price = r.operating_hours, ec.ELECTRICITY_AUD_PER_MWH / 1000.0
            heat_hours = r.preheated_hours      # heat side runs only with flue gas
            is_ro = r.technology.upper() != "MED"
            mem = (r.capex_water_aud * ec.OANDM_RATE_MED_RO
                   * ec.MEMBRANE_SHARE_OF_RO_OPEX) if is_ro else 0.0
            rows_opex.append({
                "cost_basis": basis, "architecture": r.architecture, "track_c": r.track_c,
                "om_heat_aud_yr": round(r.capex_heat_recovery_aud * ec.OANDM_RATE_ORC_HX),
                "om_desal_ex_membrane_aud_yr": round(
                    r.capex_water_aud * ec.OANDM_RATE_MED_RO - mem),
                "membrane_aud_yr": round(mem),
                "elec_id_fan_aud_yr": round(r.fan_electric_kw * heat_hours * price),
                "elec_loop_pump_aud_yr": round(r.loop_pump_kw * heat_hours * price),
                "elec_process_aud_yr": round(r.process_kw * hours * price),
                "elec_orc_credit_aud_yr": -round(r.orc_mean_kw * heat_hours * price),
                "total_opex_aud_yr": round(r.annual_opex_aud),
                "operating_hours": round(hours),
                "sec_effective_kwh_m3": round(r.sec_effective_kwh_m3, 4),
            })

    pd.DataFrame(rows_break).to_csv(OUT / "report_lcow_breakdown.csv", index=False)
    pd.DataFrame(rows_capex).to_csv(OUT / "report_capex.csv", index=False)
    pd.DataFrame(rows_opex).to_csv(OUT / "report_opex.csv", index=False)

    summ = []
    for basis, df in frames.items():
        s = df[["architecture", "track_c", "water_m3_per_day", "water_m3_per_year",
                "design_heat_kw", "hx_area_m2", "fan_kw", "fan_electric_kw", "loop_pump_kw",
                "orc_gross_kw", "orc_mean_kw", "process_kw", "net_electricity_kw",
                "sec_effective_kwh_m3",
                "operating_hours", "total_capex_aud", "annual_opex_aud",
                "capital_component_aud_per_m3", "opex_component_aud_per_m3",
                "electricity_component_aud_per_m3", "lcow_aud_per_m3"]].copy()
        s.insert(0, "cost_basis", basis)
        summ.append(s)
    pd.concat(summ).sort_values(["cost_basis", "lcow_aud_per_m3"]).to_csv(
        OUT / "report_architecture_summary.csv", index=False)

    print("wrote 4 report tables to results/")
    b = pd.DataFrame(rows_break)
    b = b[b.cost_basis == "bhojwani_global"]
    # Select each architecture's BEST track first, then pivot. Pivoting with
    # an aggregate over tracks would blend designs that do not coexist.
    best_track = (b.groupby("architecture")["lcow_total_aud_per_m3"].min().rename("best"))
    b = b.merge(best_track, on="architecture")
    b = b[np.isclose(b.lcow_total_aud_per_m3, b.best)]
    piv = b.pivot_table(index="component", columns="architecture",
                        values="aud_per_m3", aggfunc="sum")
    keep = ["Conventional SWRO (no heat recovery)", "Direct-RO (no ORC)", "orc_ro_current (path A)"]
    piv = piv[[c for c in keep if c in piv.columns]]
    print("\nLCOW DECOMPOSITION, A$/m3, bhojwani_global (best track per architecture)\n")
    print(piv.round(4).to_string())
    print("\n" + "-" * 78)
    print(piv.sum().round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
