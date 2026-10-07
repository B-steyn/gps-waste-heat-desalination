"""
Energy, heat and exergy tables.

Writes four tables to outputs/, each read from the model's own results and
functions, none typed:

  fan_power_by_design.csv          (a) added ID fan power per design: design
                                       record and mean over the operating records
  heat_by_track.csv                (b) heat collected and delivered per track: the
                                       design record and the spread over the records
  grid_import_by_architecture.csv  (c) grid electricity per m3, split into the
                                       desalination process and the heat side
  exergy_balance_designs.csv       (d) design-record exergy balance of the four
                                       reported designs, component by component

The exergy balance follows the model's conventions:
  - dead state 25 C (T_AMBIENT_CELSIUS); flue gas at constant cp over its span
    (common/exergy_analysis.gas_exergy_and_entropy_change);
  - loop water at 10 bar from CoolProp; the loop carries the full gas-side duty
    and loses 1.5 % of it (HX_SHELL_EFFICIENCY) along the way, destroyed at the
    loop's mean thermodynamic temperature;
  - the ORC from common/orc_cycle.solve_orc_cycle at the optimiser's operating
    point, which must reproduce the optimiser's net output;
  - the RO feed is seawater heated from the median intake to 38 C; the MED first
    effect is brine boiling at TBT, an isothermal sink;
  - added fan and loop pump work enter as electricity and are wholly destroyed.
It is a design-record balance: no state at the design record depends on how
the energy is charged over a year.

Run after Module 6 (run_all.py does); reporting/collect_results.py copies the
four tables into results/.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module5_orc"))

import common.constants as C                                   # noqa: E402
import common.economic_constants as ec                         # noqa: E402
from common.exergy_analysis import gas_exergy_and_entropy_change   # noqa: E402
from common.gas_properties import dynamic_flue_gas_cp          # noqa: E402
from common.orc_cycle import solve_orc_cycle                   # noqa: E402

OUT = ROOT / "outputs"
DATA = ROOT / "data"
T0 = C.T_AMBIENT_CELSIUS
T0K = T0 + 273.15
F_ORC = "R1234ze(E)"          # the reported ORC fluid (module5_orc.path_b_transcritical.ORC_TRANS_FLUID)


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


m6 = _load("m6", "module6_lcow/module6_lcow.py")
g = pd.read_csv(OUT / "gps_module6_lcow_bhojwani_global_lcow.csv")
opt = pd.read_csv(OUT / "loop_optimisation_summary_lcow.csv")
opt = opt[opt.duct.str.contains("MEASURED")]
design = m6._load_design_row_cached(str(DATA))
rec = m6._record_factors()

CONV, DRO, ORC = "Conventional SWRO (no heat recovery)", "Direct-RO (no ORC)", "orc_ro_current (path A)"
DMED, OMED = "Direct-MED", "orc_med (path A)"
NICE = {CONV: "Conventional SWRO", DRO: "Direct-RO", ORC: "ORC+RO", DMED: "Direct-MED", OMED: "ORC+MED"}


def best(arch):
    x = g[g.architecture == arch]
    return x.loc[x.lcow_aud_per_m3.idxmin()]


REPORTED = {a: best(a) for a in (CONV, DRO, ORC, DMED, OMED)}


def gas_side(track_c):
    """Design-record gas-side duty (kW, both trains) and mean cp, exactly as Module 6 sizes it."""
    cp = dynamic_flue_gas_cp((design["t_in_c"] + track_c) / 2.0, design["mole_fractions"])
    m = design["m_dot_fg"] * (C.N_TRAINS_FITTED / C.N_GAS_TRAINS)
    return m * cp * (design["t_in_c"] - track_c), m, cp


# --------------------------------------------------------------------------
# (a) added ID fan power per design
# --------------------------------------------------------------------------
def fan_table():
    rows = []
    for a in (DRO, ORC, DMED, OMED):
        r = REPORTED[a]
        mean_shaft = float(r.fan_kw) * rec["fan"]
        rows.append({
            "architecture": NICE[a], "track_c": float(r.track_c),
            "design_record_shaft_kw": float(r.fan_kw),
            "design_record_shaft_per_fan_kw": float(r.fan_kw) / C.N_TRAINS_FITTED,
            "mean_shaft_kw": mean_shaft,
            "mean_electric_kw": float(r.fan_electric_kw),
            "mean_as_pct_of_design": 100.0 * rec["fan"],
            "annual_mwh": float(r.fan_electric_kw) * float(r.preheated_hours) / 1000.0,
            "aud_per_m3": (float(r.fan_electric_kw) * float(r.preheated_hours) / 1000.0
                           * ec.ELECTRICITY_AUD_PER_MWH / float(r.water_m3_per_year)),
        })
    assert all(np.isclose(x["mean_electric_kw"], x["mean_shaft_kw"] / C.ID_FAN_MOTOR_EFFICIENCY)
               for x in rows), "fan electricity does not follow from the shaft power"
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# (b) heat collected and delivered per track
# --------------------------------------------------------------------------
def heat_table():
    t_sw, gas = m6._seawater_rows(str(DATA))
    frac = gas / gas.max()
    heat_h = float(REPORTED[DRO].preheated_hours)
    rows = []
    for t in sorted(C.TARGET_TEMPERATURES):
        q, m, cp = gas_side(t)
        q_del = q * C.HX_SHELL_EFFICIENCY
        ex, _, _ = gas_exergy_and_entropy_change(m, cp, design["t_in_c"], t)
        per = q_del * frac                  # the RO convention: heat follows gas flow
        rows.append({
            "track_c": t, "gas_side_design_kw": q, "delivered_design_kw": q_del,
            "exergy_of_recovered_heat_design_kw": ex,
            "delivered_min_kw": float(per.min()), "delivered_median_kw": float(np.median(per)),
            "delivered_mean_kw": float(per.mean()), "delivered_max_kw": float(per.max()),
            "delivered_annual_mwh": float(per.mean()) * heat_h / 1000.0,
            "n_records": len(per),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# (c) grid electricity per m3
# --------------------------------------------------------------------------
def grid_table():
    rows = []
    for a, r in REPORTED.items():
        wy = float(r.water_m3_per_year)
        process = float(r.process_kw) * float(r.operating_hours) / wy
        heat = -float(r.heat_side_kw) * float(r.preheated_hours) / wy
        total = -float(r.net_electricity_kw) * float(r.operating_hours) / wy
        assert np.isclose(total, process + heat), a
        rows.append({"architecture": NICE[a], "track_c": r.track_c,
                     "process_kwh_m3": process, "heat_side_kwh_m3": heat, "grid_import_kwh_m3": total,
                     "vs_conventional_swro_pct": np.nan})
    df = pd.DataFrame(rows)
    base = float(df.loc[df.architecture == NICE[CONV], "grid_import_kwh_m3"].iloc[0])
    df["vs_conventional_swro_pct"] = 100.0 * (df.grid_import_kwh_m3 / base - 1.0)
    return df


# --------------------------------------------------------------------------
# (d) design-record exergy balances
# --------------------------------------------------------------------------
def _water(t_c, p=C.P_WATER_LOOP_PA):
    return (CP.PropsSI("H", "T", t_c + 273.15, "P", p, "Water"),
            CP.PropsSI("S", "T", t_c + 273.15, "P", p, "Water"))


def _seawater_exergy(q_kw, t_in_c, t_out_c):
    """Exergy gained by seawater heated q_kw from t_in to t_out at constant cp."""
    m_cp = q_kw / (t_out_c - t_in_c)
    return m_cp * ((t_out_c - t_in_c) - T0K * np.log((t_out_c + 273.15) / (t_in_c + 273.15)))


def exergy_balance(arch):
    r = REPORTED[arch]
    track = float(r.track_c)
    q, m, cp = gas_side(track)
    ex_gas, _, _ = gas_exergy_and_entropy_change(m, cp, design["t_in_c"], track)
    # reference: the gas's whole sensible exergy, cooled to the dead state, with cp
    # taken over that span (condensation below the ~39 C water dew point ignored)
    cp_all = dynamic_flue_gas_cp((design["t_in_c"] + T0) / 2.0, design["mole_fractions"])
    ex_entering, _, _ = gas_exergy_and_entropy_change(m, cp_all, design["t_in_c"], T0)

    # loop: carries the full gas-side duty from return to supply, loses 1.5 % on the way
    t_sup, t_ret = track - C.LOSS_BUFFER_DELTA_T, float(r.loop_return_c)
    (h_s, s_s), (h_r, s_r) = _water(t_sup), _water(t_ret)
    m_loop = q / ((h_s - h_r) / 1000.0)
    ex_loop = m_loop * ((h_s - h_r) - T0K * (s_s - s_r)) / 1000.0
    t_lm = (h_s - h_r) / (s_s - s_r)                             # mean thermodynamic temperature, K
    q_loss = q * (1.0 - C.HX_SHELL_EFFICIENCY)
    ed_loss = q_loss * (1.0 - T0K / t_lm)
    ex_to_downstream = ex_loop - ed_loss
    q_del = q * C.HX_SHELL_EFFICIENCY

    b = {"architecture": NICE[arch], "track_c": track, "loop_return_c": t_ret, "loop_supply_c": t_sup,
         "heat_recovered_kw": q, "exergy_entering_kw": ex_entering, "exergy_recovered_kw": ex_gas,
         "ed_fgche_kw": ex_gas - ex_loop, "ed_loop_loss_kw": ed_loss,
         "ed_preheater_kw": 0.0, "ed_orc_evaporator_kw": 0.0, "ed_orc_turbine_kw": 0.0,
         "ed_orc_pump_kw": 0.0, "orc_generator_aux_loss_kw": 0.0, "ed_orc_condenser_kw": 0.0,
         "ed_med_first_effect_kw": 0.0, "orc_net_kw": 0.0, "exergy_delivered_kw": 0.0,
         "fan_work_design_kw": float(r.fan_kw), "loop_pump_work_kw": float(r.loop_pump_kw)}

    t_sw = float(np.percentile(m6._seawater_rows(str(DATA))[0], m6.SEAWATER_DESIGN_PERCENTILE))
    ceiling = C.RO_CEILING_CURRENT_MEMBRANE_C

    if arch == DRO:
        ex_feed = _seawater_exergy(q_del, t_sw, ceiling)
        b.update(ed_preheater_kw=ex_to_downstream - ex_feed, exergy_delivered_kw=ex_feed)
    elif arch == DMED:
        tbt = float(r.tbt_c)
        ex_eff = q_del * (1.0 - T0K / (tbt + 273.15))
        b.update(ed_med_first_effect_kw=ex_to_downstream - ex_eff, exergy_delivered_kw=ex_eff)
    else:                                                        # ORC+RO, ORC+MED
        if arch == ORC:
            o = opt[(opt.architecture == "orc_ro_current") & (opt.track_c == track) & (opt.path == "A")].iloc[0]
        else:
            o = opt[(opt.architecture == f"orc_med_tc{int(r.tc_c)}") & (opt.track_c == track)
                    & (opt.path == "A")]
            o = (o if len(o) else opt[(opt.architecture == "orc_med") & (opt.track_c == track)
                                      & (opt.path == "A")]).iloc[0]
        cyc = solve_orc_cycle(float(o.operating_point_c), float(o.tc_c), q_del, fluid=F_ORC)
        assert np.isclose(cyc["w_net_kw"], float(o.w_net_kw), rtol=1e-6), \
            f"{arch}: cycle {cyc['w_net_kw']:.3f} kW does not reproduce the optimiser's {o.w_net_kw:.3f}"
        sp, mw = cyc["state_points"], cyc["m_dot_wf_kg_s"]
        ex = {k: (sp[k]["h"] - T0K * sp[k]["s"]) / 1000.0 for k in sp}      # flow exergy + const, kJ/kg
        gain_evap = mw * (ex[3] - ex[2])
        released_cond = mw * (ex[4] - ex[1])
        if arch == ORC:
            ex_sink = _seawater_exergy(cyc["q_cond_kw"], t_sw, ceiling)
        else:
            tbt = float(r.tbt_c)
            ex_sink = cyc["q_cond_kw"] * (1.0 - T0K / (tbt + 273.15))
        b.update(ed_orc_evaporator_kw=ex_to_downstream - gain_evap,
                 ed_orc_turbine_kw=T0K * mw * (sp[4]["s"] - sp[3]["s"]) / 1000.0,
                 ed_orc_pump_kw=T0K * mw * (sp[2]["s"] - sp[1]["s"]) / 1000.0,
                 orc_generator_aux_loss_kw=cyc["w_turbine_kw"] - cyc["w_pump_kw"] - cyc["w_net_kw"],
                 ed_orc_condenser_kw=released_cond - ex_sink,
                 orc_net_kw=cyc["w_net_kw"], exergy_delivered_kw=ex_sink)

    parts = ["ed_fgche_kw", "ed_loop_loss_kw", "ed_preheater_kw", "ed_orc_evaporator_kw",
             "ed_orc_turbine_kw", "ed_orc_pump_kw", "orc_generator_aux_loss_kw", "ed_orc_condenser_kw",
             "ed_med_first_effect_kw", "orc_net_kw", "exergy_delivered_kw"]
    closure = b["exergy_recovered_kw"] - sum(b[k] for k in parts)
    assert abs(closure) < 1e-6 * b["exergy_recovered_kw"], f"{arch}: exergy balance open by {closure:.4f} kW"
    destroyed = [k for k in parts if k.startswith("ed_")]
    assert all(b[k] >= -1e-9 for k in destroyed), f"{arch}: negative exergy destruction"
    b["ed_heat_chain_total_kw"] = sum(b[k] for k in destroyed)
    useful = b["exergy_delivered_kw"] + b["orc_net_kw"]
    b["useful_exergy_kw"] = useful
    b["useful_pct_of_recovered"] = 100.0 * useful / b["exergy_recovered_kw"]
    b["useful_pct_of_recovered_plus_work"] = 100.0 * useful / (
        b["exergy_recovered_kw"] + b["fan_work_design_kw"] + b["loop_pump_work_kw"])
    return b


def main():
    fan = fan_table()
    heat = heat_table()
    grid = grid_table()
    exb = pd.DataFrame([exergy_balance(a) for a in (DRO, ORC, DMED, OMED)])
    fan.to_csv(OUT / "fan_power_by_design.csv", index=False)
    heat.to_csv(OUT / "heat_by_track.csv", index=False)
    grid.to_csv(OUT / "grid_import_by_architecture.csv", index=False)
    exb.to_csv(OUT / "exergy_balance_designs.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("(a) added ID fan power\n", fan.round(1).to_string(index=False))
        print("\n(b) heat by track\n", heat.round(0).to_string(index=False))
        print("\n(c) grid import\n", grid.round(3).to_string(index=False))
        print("\n(d) exergy balances (kW, design record)\n", exb.set_index("architecture").T.round(1).to_string())
    print("wrote fan_power_by_design, heat_by_track, grid_import_by_architecture, exergy_balance_designs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
