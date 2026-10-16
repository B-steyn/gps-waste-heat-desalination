"""
Supporting calculations, numbered CR4-CR15 for reference. None changes the
model: each re-evaluates existing machinery at points the model does not
report, or tabulates values it already computes - and each first reproduces
the model's own design point from scratch and fails loudly if it cannot.

  CR4   asset life by pool, corrosion and closure cases -> asset_life_sensitivity.csv
  CR5   loop-return scan, ORC+RO 85 C, paths A and B  -> loop_return_scan.csv        (figure 4)
  CR6   evaporator composite curves, invalid/corrected -> pinch_composite_curves.csv,
                                                          pinch_exergy_check.csv      (figure 3)
  CR6b  pinch-check resolution, every ORC design     -> pinch_resolution_check.csv,
                                                          pinch_resolution_profile.csv
  CR7   ORC specific cost against 2,000-5,000 A$/kW   -> orc_specific_cost_check.csv
  CR9   operating-data characterisation               -> operating_data_characterisation.csv
  CR10  price of retreating from the corrosion peak   -> corrosion_retreat_penalty.csv
  CR12  seawater temperature, 20-30 C and annualised  -> seawater_sensitivity.csv
  MED   plant sized at P50-P100 of the per-record heat -> med_sizing_percentile_check.csv
  CR15  subcritical ORC (path A) at every track       -> subcritical_orc_all_tracks.csv

Key numbers from every calculation -> supporting_calculations_summary.txt

Reads the outputs of the earlier run_all.py steps.
"""

import importlib.util
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module5_orc"))

from common.constants import (
    N_GAS_TRAINS, N_TRAINS_FITTED, LOSS_BUFFER_DELTA_T, HX_SHELL_EFFICIENCY,
    PITCH_RATIO, DUCT_WIDTH_M, DUCT_HEIGHT_M, P_WATER_LOOP_PA, ORC_EVAPORATOR_PINCH_C,
    ORC_RO_TC_CURRENT_MEMBRANE_C, RO_CEILING_CURRENT_MEMBRANE_C,
    SEAWATER_CP_KJ_KGK, RO_RECOVERY, T_AMBIENT_CELSIUS, P_AMBIENT_KPA,
)
from common.costing import area_from_pinch_profile_m2
from common.economic_constants import (
    ORC_SPECIFIC_COST_AUD_PER_KW, ASSET_LIFE_WATER_YEARS, ASSET_LIFE_HEAT_RECOVERY_YEARS,
    ELECTRICITY_AUD_PER_MWH, GPS_AVAILABILITY,
)
from common.fgche_sizing import size_fgche
from common.gas_properties import dynamic_flue_gas_cp
from common.hx_loop_physics import duct_geometry, gas_side_heat_transfer
from common.lcow import levelised_cost_of_water, annual_operating_hours
from common.loop_optimisation import evaluate_loop_return
from common.pinch_check import discretized_pinch_check
from common.ro_thermal import ro_specific_energy


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m6 = _load("m6", "module6_lcow/module6_lcow.py")

TRACK_C = 85.0                          # the ORC+RO design track (FINAL_RESULTS 3_design)
TC_C = ORC_RO_TC_CURRENT_MEMBRANE_C     # 38 C membrane ceiling + 5 K condenser pinch
CEILING_C = RO_CEILING_CURRENT_MEMBRANE_C
BASES = ("bhojwani_global",)
SUMMARY = []


def note(line=""):
    print(line, flush=True)
    SUMMARY.append(line)


def _m6(basis):
    return pd.read_csv(OUT / f"gps_module6_lcow_{basis}_lcow.csv")


def _opt():
    o = pd.read_csv(OUT / "loop_optimisation_summary_lcow.csv")
    return o[o["duct"].str.contains("MEASURED")]


def _design_entry(path="A", arch="orc_ro_current", track=TRACK_C):
    o = _opt()
    return o[(o.architecture == arch) & (o.track_c == track) & (o.path == path)].iloc[0]


def _close(a, b, rel=1e-6, what=""):
    if not np.isclose(float(a), float(b), rtol=rel, atol=1e-9):
        raise AssertionError(f"{what}: {a!r} does not reproduce {b!r}")


def _sec(t):
    return ro_specific_energy(t, r=RO_RECOVERY)["sec_kwh_m3"]


# --------------------------------------------------------------------------
# ORC+RO assembly - module6_lcow.run's own call, applied to one design entry
# --------------------------------------------------------------------------
def _assemble_orc_ro(entry, anchor, water, sec_pre, sec_amb, sink_in):
    return m6.assemble_architecture(
        orc_evap_ua_w_per_k=float(entry["orc_evap_ua_w_per_k"]),
        orc_t_cond_c=float(entry["tc_c"]), orc_sink_in_c=sink_in,
        orc_sink_out_c=CEILING_C, name="orc_ro_current", technology="SWRO",
        design_heat_kw=float(entry["q_cond_kw"]), water_m3_per_day=water,
        hx_area_m2=float(entry["area_total_m2"]), fan_kw=float(entry["fan_kw"]),
        loop_pump_kw=float(entry["loop_pump_kw"]), orc_gross_kw=float(entry["w_net_kw"]),
        process_sec_kwh_m3=sec_pre, sec_ambient_kwh_m3=sec_amb, anchor=anchor)


def orc_ro_lcow(entry, anchor):
    perf = m6.ro_annual_performance(float(entry["q_cond_kw"]), CEILING_C, str(DATA))
    if perf is None:
        return None, None
    rec = _assemble_orc_ro(entry, anchor, perf["water_m3_per_day"], perf["sec_preheated_kwh_m3"],
                           perf["sec_ambient_kwh_m3"], perf["design_seawater_c"])
    return rec, perf


# --------------------------------------------------------------------------
# The optimiser's own set-up at the measured duct and the 85 C track
# (module5_orc.loop_optimiser.run_case), built once per process
# --------------------------------------------------------------------------
_CTX = {}


def _context():
    if not _CTX:
        from design_cases import load_design_row
        d = load_design_row(str(DATA))
        n_tubes, a_flow = duct_geometry(DUCT_WIDTH_M, PITCH_RATIO)
        gt = gas_side_heat_transfer(d["t_in_c"], d["m_dot_fg"], d["p_gauge"], d["dust"], a_flow)
        cp_d = dynamic_flue_gas_cp((d["t_in_c"] + TRACK_C) / 2.0, d["mole_fractions"])
        q_target = d["m_dot_fg"] * (N_TRAINS_FITTED / N_GAS_TRAINS) * cp_d * (d["t_in_c"] - TRACK_C)
        _CTX.update(design=d, n_tubes=n_tubes, gt=gt, q_target=q_target,
                    q_in=q_target * HX_SHELL_EFFICIENCY,
                    c_gas=(d["m_dot_fg"] / N_GAS_TRAINS) * cp_d,
                    t_hot=TRACK_C - LOSS_BUFFER_DELTA_T)
    return _CTX


def _evaluate(path, lr):
    from loop_optimiser import _solver_a, _solver_b
    c = _context()
    solver = partial(_solver_a if path == "A" else _solver_b,
                     q_in=c["q_in"], tc=TC_C, t_hot=c["t_hot"])
    return evaluate_loop_return(float(lr), lambda hh, hr, md: solver(hh, hr, md),
                                c["t_hot"], c["q_target"], c["design"]["t_in_c"], c["c_gas"],
                                c["gt"], c["n_tubes"], DUCT_HEIGHT_M)


def _scan_point(job):
    path, lr = job
    e = _evaluate(path, lr)
    if e is None:
        return {"path": path, "loop_return_c": lr, "feasible": False}
    return {"path": path, "loop_return_c": lr, "feasible": True, "rows": e["rows"],
            "min_pinch_k": e["pinch"]["min_pinch_k"], "orc_t_evap_c": e["operating_point_c"],
            "w_net_kw": e["w_net_kw"], "fan_kw": e["fan_kw"], "loop_pump_kw": e["loop_pump_kw"],
            # annual mean over the operating records, as the LCOW charges it
            "net_electricity_kw": m6.heat_side_annual_kw(e["w_net_kw"], e["fan_kw"],
                                                         e["loop_pump_kw"])["heat_side_kw"],
            "net_electricity_design_kw": e["w_net_kw"] - e["fan_kw"] - e["loop_pump_kw"],
            "area_total_m2": e["area_total_m2"], "q_cond_kw": e["q_cond_kw"],
            "m_dot_loop_kg_s": e["m_dot_loop_kg_s"], "tc_c": TC_C,
            "orc_evap_ua_w_per_k": area_from_pinch_profile_m2(
                e["pinch"]["profile"], e["pinch"]["q_total_w"], 1.0)}


# ==========================================================================
# CR5 - loop-return scan (figure F4)
# ==========================================================================
def cr5(pool):
    note("== CR5 loop-return scan (ORC+RO, 85 C track, measured duct)")
    design_lr = {p: float(_design_entry(p).loop_return_c)
                 for p in ("A", "B") if not _opt().query(
                     "architecture == 'orc_ro_current' and track_c == @TRACK_C and path == @p").empty}
    c_upper = min(TRACK_C - LOSS_BUFFER_DELTA_T - 1.0, TRACK_C - 5.0)  # the optimiser's own bounds
    grid = np.round(np.arange(TC_C + 2.0, c_upper + 1e-9, 0.5), 3)
    jobs = [(p, float(lr)) for p in ("A", "B") for lr in grid] + list(design_lr.items())
    df = pd.DataFrame(pool.map(_scan_point, jobs))

    cap = m6._MED_ROW_CAP
    df["fan_feasible"] = df["feasible"] & (df["rows"].fillna(1e9) <= cap)
    df["is_model_design"] = [p in design_lr and abs(lr - design_lr[p]) < 1e-9
                             for p, lr in zip(df.path, df.loop_return_c)]
    for basis, col in (("bhojwani_global", "lcow_aud_per_m3"),):
        vals, water = [], []
        for _, r in df.iterrows():
            if not r.feasible:
                vals.append(np.nan); water.append(np.nan); continue
            rec, perf = orc_ro_lcow(r, basis)
            vals.append(rec["lcow_aud_per_m3"] if rec else np.nan)
            water.append(perf["water_m3_per_day"] if perf else np.nan)
        df[col] = vals
        df["water_m3_per_day"] = water
    df = df.sort_values(["path", "loop_return_c"]).reset_index(drop=True)

    # reproduction: the scan at the optimiser's own loop return must give Module 6's LCOW
    m6_row = _m6("bhojwani_global").query(
        "architecture == 'orc_ro_current (path A)' and track_c == @TRACK_C").iloc[0]
    d_a = df[(df.path == "A") & df.is_model_design].iloc[0]
    _close(d_a.lcow_aud_per_m3, m6_row.lcow_aud_per_m3, what="CR5 design point LCOW")
    _close(d_a.area_total_m2, _design_entry("A").area_total_m2, what="CR5 design point area")
    note(f"  reproduces Module 6: path A at {d_a.loop_return_c:.3f} C -> "
         f"A${d_a.lcow_aud_per_m3:.3f}/m3, {d_a.area_total_m2:,.0f} m2, {int(d_a.rows)} rows")

    for p in ("A", "B"):
        s = df[(df.path == p) & df.fan_feasible].sort_values("loop_return_c")
        if s.empty:
            note(f"  path {p}: no fan-feasible point"); continue
        lo, hi = s.loop_return_c.min(), s.loop_return_c.max()
        best = s.loc[s.lcow_aud_per_m3.idxmin()]
        ne = s.loc[s.net_electricity_kw.idxmax()]
        wn = s.loc[s.w_net_kw.idxmax()]
        rises = int((np.diff(s.lcow_aud_per_m3.values) < -1e-9).sum())
        note(f"  path {p}: fan-feasible loop return {lo:.2f}-{hi:.2f} C "
             f"({len(s)} points; infeasible below the cold-end pinch, above by the 33-row fan cap)")
        note(f"    lowest LCOW  A${best.lcow_aud_per_m3:.3f}/m3 at {best.loop_return_c:.2f} C"
             f"  (LCOW falls as loop return rises at {rises} of {len(s) - 1} steps)")
        note(f"    highest net electricity {ne.net_electricity_kw:,.0f} kW at {ne.loop_return_c:.2f} C;"
             f" highest ORC output {wn.w_net_kw:,.1f} kW at {wn.loop_return_c:.2f} C")
    for p in ("A", "B"):
        s = df[(df.path == p) & df.feasible]
        if s.empty:
            continue
        note(f"  path {p} ignoring the fan cap ({s.loop_return_c.min():.2f}-{s.loop_return_c.max():.2f} C, "
             f"{int(s.rows.min())}-{int(s.rows.max())} rows): lowest LCOW at "
             f"{s.loc[s.lcow_aud_per_m3.idxmin()].loop_return_c:.2f} C, highest net electricity "
             f"{s.net_electricity_kw.max():,.0f} kW at {s.loc[s.net_electricity_kw.idxmax()].loop_return_c:.2f} C, "
             f"highest ORC output {s.w_net_kw.max():,.0f} kW at {s.loc[s.w_net_kw.idxmax()].loop_return_c:.2f} C")

    cols = ["loop_return_c", "path", "feasible", "fan_feasible", "is_model_design", "rows",
            "min_pinch_k", "orc_t_evap_c", "w_net_kw", "net_electricity_kw", "net_electricity_design_kw", "fan_kw",
            "loop_pump_kw", "area_total_m2", "q_cond_kw", "water_m3_per_day",
            "lcow_aud_per_m3"]
    df[cols].to_csv(OUT / "loop_return_scan.csv", index=False)
    return df


# ==========================================================================
# CR6 - evaporator composite curves (figure F3) and the exergy inset
# ==========================================================================
def _composite(case, cycle, sized, fluid, n, phase=True):
    sp = cycle["state_points"]
    kw = dict(hot_fluid="Water", p_hot_pa=P_WATER_LOOP_PA, h_hot_in=sized["h_loop_hot"],
              h_hot_out=sized["h_loop_return"], m_dot_hot=sized["m_dot_loop_kg_s"],
              cold_fluid=fluid, p_cold_pa=cycle.get("p_evap_pa", cycle.get("p_high_pa")), h_cold_in=sp[2]["h"],
              h_cold_out=sp[3]["h"], m_dot_cold=cycle["m_dot_wf_kg_s"])
    # phase=False gives the evenly spaced check alone (no bubble or dew points)
    pc = discretized_pinch_check(**kw, n_segments=n, include_phase_change_points=phase)
    q_h = kw["m_dot_hot"] * (kw["h_hot_in"] - kw["h_hot_out"])
    q_c = kw["m_dot_cold"] * (kw["h_cold_out"] - kw["h_cold_in"])
    prof = pd.DataFrame(pc["profile"]).rename(columns={"duty_fraction": "fraction"})
    prof.insert(0, "case", case)
    prof["q_cumulative_kw"] = prof["fraction"] * q_h / 1000.0

    # Gouy-Stodola along the exchanger: E_D = T0 * S_gen >= 0 in every segment of
    # a real exchanger. The hot stream runs hot end -> cold end, the working fluid
    # the other way; the 1.5 % shell loss leaves to ambient at T0.
    t0 = T_AMBIENT_CELSIUS + 273.15
    f = prof["fraction"].values
    s_h = [CP.PropsSI("S", "P", kw["p_hot_pa"], "H", kw["h_hot_in"] - x * q_h / kw["m_dot_hot"], "Water")
           for x in f]
    s_c = [CP.PropsSI("S", "P", kw["p_cold_pa"], "H", kw["h_cold_out"] - x * q_c / kw["m_dot_cold"], fluid)
           for x in f]
    s_gen = [kw["m_dot_hot"] * (s_h[i + 1] - s_h[i]) + kw["m_dot_cold"] * (s_c[i] - s_c[i + 1])
             + (q_h - q_c) * (f[i + 1] - f[i]) / t0
             for i in range(len(f) - 1)]          # slices are uneven (bubble and dew points)
    ex = pd.DataFrame({"case": case, "fraction": f,
                       "e_d_cumulative_kw": np.concatenate([[0.0], np.cumsum(s_gen) * t0 / 1000.0])})
    return prof, ex, pc


def cr6():
    note("== CR6 pinch composite curves (ORC+RO design point, path A evaporator)")
    from path_b_transcritical import ORC_TRANS_FLUID      # the fluid _solver_a passes
    c = _context()
    lr = float(_design_entry("A").loop_return_c)
    e = _evaluate("A", lr)
    sized = size_fgche(lr, c["t_hot"], c["q_target"], c["design"]["t_in_c"], c["c_gas"],
                       c["gt"], c["n_tubes"], DUCT_HEIGHT_M, p_loop_pa=P_WATER_LOOP_PA)
    _close(sized["m_dot_loop_kg_s"], e["m_dot_loop_kg_s"], what="CR6 loop flow")
    res = e["result"]
    cases = {"invalid": res["unconstrained"], "corrected": res["constrained"]}

    # reproduction at the model's own 20 segments
    _, _, pc20 = _composite("corrected", cases["corrected"]["cycle"], sized, ORC_TRANS_FLUID, 20)
    _close(pc20["min_pinch_k"], e["pinch"]["min_pinch_k"], what="CR6 corrected min pinch (20 seg)")

    curves, exer = [], []
    for case, sol in cases.items():
        prof, ex, pc = _composite(case, sol["cycle"], sized, ORC_TRANS_FLUID, 100)
        prof["t_evap_c"] = sol["t_evap_c"]
        prof["endpoint_hot_k"] = prof["approach_k"].iloc[0]
        prof["endpoint_cold_k"] = prof["approach_k"].iloc[-1]
        curves.append(prof); exer.append(ex)
        note(f"  {case:9s}: T_evap {sol['t_evap_c']:.2f} C, W {sol['cycle']['w_net_kw']:.1f} kW, "
             f"min approach {pc['min_pinch_k']:+.2f} K at {pc['pinch_location_fraction']:.2f} of duty; "
             f"endpoints hot {prof.approach_k.iloc[0]:+.2f} K / cold {prof.approach_k.iloc[-1]:+.2f} K; "
             f"hot end to saturation {c['t_hot'] - sol['t_evap_c']:.2f} K; "
             f"exergy destroyed {ex.e_d_cumulative_kw.iloc[-1]:+.1f} kW "
             f"(min along the exchanger {ex.e_d_cumulative_kw.min():+.1f} kW)")
    curves = pd.concat(curves, ignore_index=True)
    exer = pd.concat(exer, ignore_index=True)
    inv = curves[curves.case == "invalid"].approach_k.min()
    cor = curves[curves.case == "corrected"].approach_k.min()
    assert inv < 0, f"CR6: the invalid design should cross, min approach {inv:.2f} K"

    # resolution: the model checks 21 points; the pinch of a boiling stream sits
    # at a kink (the bubble point) that can fall between two of them
    fine = {n: _composite("corrected", cases["corrected"]["cycle"], sized, ORC_TRANS_FLUID, n)[2]
            for n in (20, 100, 1000)}
    note("  corrected design, minimum approach by resolution: "
         + ", ".join(f"{n} segments {v['min_pinch_k']:.3f} K" for n, v in fine.items()))
    if fine[1000]["min_pinch_k"] < 5.0 - 0.05:
        note(f"  FINDING: at 1,000 segments the corrected design's approach is "
             f"{fine[1000]['min_pinch_k']:.2f} K, below the 5.0 K the 20-segment check certifies")
    note(f"  working fluid {ORC_TRANS_FLUID}; dead state {T_AMBIENT_CELSIUS} C")

    cols = ["case", "fraction", "q_cumulative_kw", "t_hot_c", "t_cold_c", "approach_k",
            "t_evap_c", "endpoint_hot_k", "endpoint_cold_k"]
    curves[cols].to_csv(OUT / "pinch_composite_curves.csv", index=False)
    exer.to_csv(OUT / "pinch_exergy_check.csv", index=False)
    return curves, exer, fine


# ==========================================================================
# CR6b - is the model's 20-segment pinch check resolved?
# ==========================================================================
def _ctx_track(track):
    """run_case's own set-up for the measured duct, at any extraction track."""
    d = _context()["design"]
    n_tubes, a_flow = duct_geometry(DUCT_WIDTH_M, PITCH_RATIO)
    gt = gas_side_heat_transfer(d["t_in_c"], d["m_dot_fg"], d["p_gauge"], d["dust"], a_flow)
    cp_d = dynamic_flue_gas_cp((d["t_in_c"] + track) / 2.0, d["mole_fractions"])
    q_target = d["m_dot_fg"] * (N_TRAINS_FITTED / N_GAS_TRAINS) * cp_d * (d["t_in_c"] - track)
    return {"design": d, "n_tubes": n_tubes, "gt": gt, "q_target": q_target,
            "q_in": q_target * HX_SHELL_EFFICIENCY,
            "c_gas": (d["m_dot_fg"] / N_GAS_TRAINS) * cp_d, "t_hot": track - LOSS_BUFFER_DELTA_T}


def cr6_resolution():
    """
    Every ORC design Module 6 reports is re-solved, and its evaporator is
    checked three independent ways:
      - the model's own 20-segment grid (must reproduce the optimiser),
      - a 1,000-segment grid,
      - exactly at the working fluid's bubble and dew points, located from
        saturation properties, with no grid at all.
    A negative minimum would be a CROSSING, which is thermodynamically
    impossible. A positive minimum below 5 K breaches the design approach only.
    """
    note("== CR6b pinch-check resolution, every ORC design Module 6 reports")
    from loop_optimiser import _solver_a, _solver_b
    from path_b_transcritical import ORC_TRANS_FLUID as FLUID
    g, opt = _m6("bhojwani_global"), _opt()
    p_crit = CP.PropsSI("Pcrit", FLUID)
    recs, profile, all_profiles = [], None, []
    for _, r in g[g.architecture.str.startswith("orc_")].iterrows():
        arch, path = r.architecture.split(" (path ")
        path = path.rstrip(")")
        key = arch
        if arch == "orc_med":
            key = "orc_med" if float(r.tc_c) == 65.0 else f"orc_med_tc{float(r.tc_c):.0f}"
        o = opt[(opt.architecture == key) & (opt.track_c == r.track_c) & (opt.path == path)].iloc[0]
        c, lr, tc = _ctx_track(float(r.track_c)), float(o.loop_return_c), float(o.tc_c)
        solver = partial(_solver_a if path == "A" else _solver_b, q_in=c["q_in"], tc=tc, t_hot=c["t_hot"])
        e = evaluate_loop_return(lr, lambda hh, hr, md: solver(hh, hr, md), c["t_hot"], c["q_target"],
                                 c["design"]["t_in_c"], c["c_gas"], c["gt"], c["n_tubes"], DUCT_HEIGHT_M)
        _close(e["w_net_kw"], o.w_net_kw, what=f"CR6b {r.architecture} {r.track_c:.0f} C")
        sized = size_fgche(lr, c["t_hot"], c["q_target"], c["design"]["t_in_c"], c["c_gas"],
                           c["gt"], c["n_tubes"], DUCT_HEIGHT_M, p_loop_pa=P_WATER_LOOP_PA)
        cyc = e["cycle"]
        m20 = _composite("x", cyc, sized, FLUID, 20)[2]["min_pinch_k"]
        _close(m20, e["pinch"]["min_pinch_k"], what=f"CR6b {r.architecture} model-check reproduction")
        m20_old = _composite("x", cyc, sized, FLUID, 20, phase=False)[2]["min_pinch_k"]
        prof1k, _, pc1k = _composite("corrected", cyc, sized, FLUID, 1000)
        p_cold = cyc.get("p_evap_pa", cyc.get("p_high_pa"))
        h_in, h_out = cyc["state_points"][2]["h"], cyc["state_points"][3]["h"]
        at = {}
        if p_cold < p_crit:                       # subcritical: a boiling plateau exists
            t_sat = CP.PropsSI("T", "P", p_cold, "Q", 0, FLUID) - 273.15
            for lab, qual in (("bubble", 0), ("dew", 1)):
                f = (h_out - CP.PropsSI("H", "P", p_cold, "Q", qual, FLUID)) / (h_out - h_in)
                h_hot = sized["h_loop_hot"] - f * (sized["h_loop_hot"] - sized["h_loop_return"])
                at[lab] = (CP.PropsSI("T", "P", P_WATER_LOOP_PA, "H", h_hot, "Water") - 273.15 - t_sat, f)
        true_min = min([pc1k["min_pinch_k"]] + [v[0] for v in at.values()])
        all_profiles.append(prof1k.assign(design=f"{r.architecture} {r.track_c:.0f} C",
                                          cold_side="subcritical" if at else "supercritical"))
        recs.append({"architecture": r.architecture, "track_c": r.track_c, "path": path, "tc_c": tc,
                     "loop_return_c": lr, "orc_output_kw": e["w_net_kw"],
                     "cold_side": "subcritical" if at else "supercritical",
                     "min_approach_model_check_k": m20,
                     "min_approach_old_even_grid_k": m20_old,
                     "min_approach_1000seg_k": pc1k["min_pinch_k"],
                     "approach_at_bubble_point_k": at.get("bubble", (np.nan,))[0],
                     "bubble_point_duty_fraction": at.get("bubble", (np.nan, np.nan))[1],
                     "approach_at_dew_point_k": at.get("dew", (np.nan,))[0],
                     "true_min_approach_k": true_min,
                     "shortfall_below_5k": max(0.0, ORC_EVAPORATOR_PINCH_C - true_min),
                     "streams_cross": bool(true_min < 0.0)})
        if r.architecture == "orc_ro_current (path A)" and float(r.track_c) == TRACK_C:
            # flag the model's own check points by position, not by index: the
            # model's grid includes the bubble and dew points
            checked = [pt["duty_fraction"] for pt in e["pinch"]["profile"]]
            profile = prof1k.assign(model_check_point=[
                any(abs(f - c) < 1e-12 for c in checked) for f in prof1k.fraction])
    # Before and after adding the bubble and dew points to the check, for
    # figure 7. "Before" is the ORC+RO design an even-grid check alone
    # certified, rebuilt from the frozen designs in validation/fixtures; its
    # check points are 21 even points.
    from common.orc_cycle import solve_orc_cycle
    from common.constants import ORC_TURBINE_ISENTROPIC_EFF, ORC_PUMP_ISENTROPIC_EFF
    frz = pd.read_csv(ROOT / "validation" / "fixtures" / "pinch_pre_fix_optimiser_designs.csv")
    frz = frz[frz.duct.str.contains("MEASURED") & (frz.architecture == "orc_ro_current")
              & (frz.track_c == TRACK_C) & (frz.path == "A")].iloc[0]
    cb = _ctx_track(TRACK_C)
    sized_b = size_fgche(float(frz.loop_return_c), cb["t_hot"], cb["q_target"], cb["design"]["t_in_c"],
                         cb["c_gas"], cb["gt"], cb["n_tubes"], DUCT_HEIGHT_M, p_loop_pa=P_WATER_LOOP_PA)
    cyc_b = solve_orc_cycle(t_evap_c=float(frz.operating_point_c), t_cond_c=float(frz.tc_c), q_in_kw=cb["q_in"],
                            fluid=FLUID, turbine_eff=ORC_TURBINE_ISENTROPIC_EFF, pump_eff=ORC_PUMP_ISENTROPIC_EFF)
    _close(cyc_b["w_net_kw"], frz.w_net_kw, what="CR6b pre-fix ORC+RO design rebuilt")
    prof_b = _composite("before the fix", cyc_b, sized_b, FLUID, 1000)[0]
    old_points = [i / 20 for i in range(21)]
    prof_b["model_check_point"] = [any(abs(f - c) < 1e-12 for c in old_points) for f in prof_b.fraction]
    ba = pd.concat([prof_b, profile.assign(case="after the fix")], ignore_index=True)
    ba["pct_duty_from_cold_end"] = (1.0 - ba.fraction) * 100.0
    ba[["case", "fraction", "pct_duty_from_cold_end", "q_cumulative_kw", "t_hot_c", "t_cold_c", "approach_k",
        "model_check_point"]].to_csv(OUT / "pinch_before_after_profile.csv", index=False)
    ap = pd.concat(all_profiles, ignore_index=True)
    ap["pct_duty_from_cold_end"] = (1.0 - ap.fraction) * 100.0
    ap[["design", "cold_side", "fraction", "pct_duty_from_cold_end", "approach_k"]].to_csv(
        OUT / "pinch_profiles_all_designs.csv", index=False)
    note(f"  ORC+RO before the fix: true minimum {prof_b.approach_k.min():.2f} K; after: "
         f"{profile.approach_k.min():.2f} K; all {ap.design.nunique()} designs >= {ap.approach_k.min():.2f} K")
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "pinch_resolution_check.csv", index=False)
    profile[["fraction", "q_cumulative_kw", "t_hot_c", "t_cold_c", "approach_k", "model_check_point"]].to_csv(
        OUT / "pinch_resolution_profile.csv", index=False)
    worst = df.loc[df.true_min_approach_k.idxmin()]
    note(f"  {len(df)} designs: streams cross in {int(df.streams_cross.sum())}; "
         f"{int((df.shortfall_below_5k > 0.05).sum())} fall short of 5 K by more than 0.05 K; worst "
         f"{worst.true_min_approach_k:.2f} K ({worst.architecture}, {worst.track_c:.0f} C)")
    for _, x in df.iterrows():
        note(f"    {x.architecture:26s} {x.track_c:4.0f} C  {x.cold_side:13s} model {x.min_approach_model_check_k:5.2f}"
             f" (old even grid {x.min_approach_old_even_grid_k:5.2f})"
             f" | 1000 seg {x.min_approach_1000seg_k:5.2f} | at bubble point {x.approach_at_bubble_point_k:5.2f} K"
             f" (at {x.bubble_point_duty_fraction:.3f} of duty)")
    return df


# ==========================================================================
# MED sizing percentile - is P90 a sensible size?
# ==========================================================================
def med_sizing_check():
    """
    The plant is sized to the p-th percentile of the per-record heat. In each
    running hour it makes water from the heat that actually arrives, capped
    at its size. Summed over the 461 recorded hours, that IS the capacity
    factor - it is the per-record data, not a reduction applied on top of it.
    """
    note("== MED sizing percentile (Direct-MED)")
    recs = []
    for basis in BASES:
        g = _m6(basis)
        x = g[g.architecture == "Direct-MED"]
        r = x.loc[x.lcow_aud_per_m3.idxmin()]
        q = pd.read_csv(OUT / f"gps_module3_output_direct_med_{int(r.track_c)}.csv")[
            "heat_delivered_to_downstream_kw"]
        q = q[q > 0].values
        p90 = float(np.percentile(q, 90))
        _close(p90, r.design_heat_kw, what="MED design heat is P90 of the per-row heat")
        _close(np.minimum(q, p90).mean() / p90, r.capacity_factor, rel=1e-9, what="MED capacity factor")
        if basis == "bhojwani_global":
            use = np.minimum(q, p90).mean()
            note(f"  {int(r.track_c)} C track, {len(q)} running rows: heat mean {q.mean():,.0f} kW, "
                 f"P90 {p90:,.0f}, max {q.max():,.0f}")
            note(f"    usable by a P90-sized plant: {use:,.0f} kW on average = {use / p90:.4f} of its size; "
                 f"spilled {100 * (1 - use / q.mean()):.2f} % of the heat")
            note(f"    nameplate {r.water_m3_per_day:,.0f} m3/day; average while running "
                 f"{r.water_m3_per_day * use / p90:,.0f} m3/day")
        for pct in (50, 60, 70, 80, 85, 90, 95, 100):
            qp = float(np.percentile(q, pct))
            cf = float(np.minimum(q, qp).mean() / qp)
            rec = m6.assemble_architecture(
                name="Direct-MED", technology="MED", design_heat_kw=qp,
                water_m3_per_day=r.water_m3_per_day * qp / r.design_heat_kw,
                hx_area_m2=r.hx_area_m2, fan_kw=r.fan_kw, loop_pump_kw=r.loop_pump_kw,
                orc_gross_kw=0.0, process_sec_kwh_m3=m6.MED_PROCESS_SEC_KWH_M3,
                med_gor=r.med_gor, capacity_factor=cf, anchor=basis)
            if pct == 90:
                _close(rec["lcow_aud_per_m3"], r.lcow_aud_per_m3, what=f"MED sizing reproduction {basis}")
            recs.append({"cost_basis": basis, "track_c": r.track_c, "sizing_percentile": pct,
                         "design_heat_kw": qp, "capacity_m3_per_day": r.water_m3_per_day * qp / r.design_heat_kw,
                         "capacity_factor": cf,
                         "heat_spilled_pct": 100.0 * (1.0 - np.minimum(q, qp).sum() / q.sum()),
                         "water_m3_per_year": rec["water_m3_per_year"],
                         "lcow_aud_per_m3": rec["lcow_aud_per_m3"]})
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "med_sizing_percentile_check.csv", index=False)
    for basis in BASES:
        s = df[df.cost_basis == basis].set_index("sizing_percentile")
        note(f"  {basis:15s}: " + ", ".join(f"P{p} {v:.3f}" for p, v in s.lcow_aud_per_m3.items())
             + f"  -> lowest at P{s.lcow_aud_per_m3.idxmin()}")
    return df


# ==========================================================================
# CR7 - ORC specific investment cost against the scope of work's range
# ==========================================================================
def cr7():
    note("== CR7 ORC specific cost against the 2,000-5,000 A$/kW scope-of-work range")
    g, opt = _m6("bhojwani_global"), _opt()
    lo, hi = ORC_SPECIFIC_COST_AUD_PER_KW
    recs = []
    for _, r in g[g.architecture.str.startswith("orc_")].iterrows():
        arch, path = r.architecture.split(" (path ")
        path = path.rstrip(")")
        key = arch
        if arch == "orc_med":
            key = "orc_med" if float(r.tc_c) == 65.0 else f"orc_med_tc{float(r.tc_c):.0f}"
        o = opt[(opt.architecture == key) & (opt.track_c == r.track_c) & (opt.path == path)]
        if o.empty or not np.isclose(o.w_net_kw.iloc[0], r.orc_gross_kw, rtol=1e-6):
            note(f"  WARNING: no matching optimiser row for {r.architecture} at {r.track_c:.0f} C - skipped")
            continue
        o = o.iloc[0]
        q_evap = o.w_net_kw + o.q_cond_kw          # cycle balance: q_in = W_net + q_cond, exact
        area_based = r.capex_orc_package_aud + r.capex_orc_exchangers_aud
        recs.append({
            "architecture": r.architecture, "track_c": r.track_c, "tc_c": o.tc_c,
            "orc_gross_kw": o.w_net_kw, "q_evap_kw": q_evap, "q_cond_kw": o.q_cond_kw,
            "orc_thermal_efficiency_pct": 100.0 * o.w_net_kw / q_evap,
            "capex_orc_package_aud": r.capex_orc_package_aud,
            "capex_orc_exchangers_aud": r.capex_orc_exchangers_aud,
            "implied_aud_per_kw_package_only": r.capex_orc_package_aud / o.w_net_kw,
            "implied_aud_per_kw_area_based": area_based / o.w_net_kw,
            "inside_literature_band": bool(lo <= area_based / o.w_net_kw <= hi)})
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "orc_specific_cost_check.csv", index=False)
    d = df.loc[(df.architecture == "orc_ro_current (path A)") & (df.track_c == TRACK_C)].iloc[0]
    note(f"  ORC+RO design: {d.orc_gross_kw:.1f} kW from {d.q_evap_kw:,.0f} kW "
         f"(thermal efficiency {d.orc_thermal_efficiency_pct:.2f} %); "
         f"A${d.implied_aud_per_kw_area_based:,.0f}/kW with its exchangers "
         f"({d.implied_aud_per_kw_area_based / hi:.0f}x the band's top)")
    note(f"  across {len(df)} ORC designs: efficiency {df.orc_thermal_efficiency_pct.min():.2f}-"
         f"{df.orc_thermal_efficiency_pct.max():.2f} %, A${df.implied_aud_per_kw_area_based.min():,.0f}-"
         f"{df.implied_aud_per_kw_area_based.max():,.0f}/kW; inside the band: "
         f"{int(df.inside_literature_band.sum())} of {len(df)}")
    return df


# ==========================================================================
# Seawater: the observed record and a full-year sinusoid through it
# ==========================================================================
def _water_records():
    w = pd.read_csv(DATA / "GPS_Water_related_data.csv")
    w.columns = w.columns.str.strip()
    w["t"] = pd.to_datetime(w["Timestamp"].astype(str).str.strip())
    return w.sort_values("t").reset_index(drop=True)


def _seawater_fit():
    """T(t) = a + b cos(wt) + c sin(wt), one-year period, least squares over every
    record. Five months of data constrain the phase and amplitude only loosely,
    so this is a crude fit; the 5 C intake bound remains the conservative
    statement."""
    w = _water_records()
    t = (w.t - pd.Timestamp(w.t.dt.year.iloc[0], 1, 1)).dt.total_seconds().values / 86400.0
    y = w["Sea water intake temp"].values
    om = 2.0 * np.pi / 365.25
    X = np.column_stack([np.ones_like(t), np.cos(om * t), np.sin(om * t)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    r2 = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    yr = np.arange(0.0, 365.0, 0.25)
    year = coef[0] + coef[1] * np.cos(om * yr) + coef[2] * np.sin(om * yr)
    return {"mean": coef[0], "amplitude": float(np.hypot(coef[1], coef[2])),
            "peak_day": float((np.arctan2(coef[2], coef[1]) / om) % 365.25),
            "r2": r2, "rmse": float(np.sqrt(np.mean((y - fitted) ** 2))),
            "window": fitted, "year": year}


# ==========================================================================
# CR9 - operating-data characterisation
# ==========================================================================
def cr9():
    note("== CR9 operating-data characterisation")
    rows = []

    def add(q, v, unit, n=""):
        rows.append({"quantity": q, "value": v, "unit": unit, "note": n})

    m1 = pd.read_csv(OUT / "GPS_Module1_Output.csv")
    x, y = m1["Unit Load (MW)"].values, m1["FF_Outlet_Temp_Avg_C"].values
    slope, icpt = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    add("(a) exhaust temperature vs load - slope", slope, "C per MW", f"least squares, {len(x)} operating rows")
    add("(a) exhaust temperature vs load - intercept", icpt, "C")
    add("(a) exhaust temperature vs load - R squared", r2, "-")
    add("(a) fitted exhaust temperature at minimum load", icpt + slope * x.min(), "C", f"{x.min():.0f} MW")
    add("(a) fitted exhaust temperature at maximum load", icpt + slope * x.max(), "C", f"{x.max():.0f} MW")
    for k, v in (("min", y.min()), ("mean", y.mean()), ("max", y.max()), ("std", y.std(ddof=1))):
        add(f"(a) observed exhaust temperature - {k}", v, "C")
    note(f"  (a) exhaust {y.min():.1f}-{y.max():.1f} C; slope {slope:.4f} C/MW, R2 {r2:.3f}; "
         f"fitted {icpt + slope * x.min():.1f} C at {x.min():.0f} MW -> {icpt + slope * x.max():.1f} C "
         f"at {x.max():.0f} MW")

    w = _water_records()
    gap = w.t.diff().dropna() / pd.Timedelta(hours=1)
    spacing = float(gap.mode().iloc[0])
    down = (w["Unit Load (MW)"] < 1.0).values
    add("(b) raw records", len(w), "-", f"{w.t.min():%Y-%m-%d} to {w.t.max():%Y-%m-%d}")
    add("(b) record spacing", spacing, "h", f"largest gap {gap.max():.0f} h")
    add("(b) non-operating records (load < 1 MW)", int(down.sum()), "-",
        f"availability {1 - down.mean():.4f}")
    runs, start = [], None
    for i, d in enumerate(down):
        contiguous = i > 0 and gap.iloc[i - 1] <= spacing + 1e-9
        if d and (start is None or not contiguous):
            if start is not None:
                runs.append((start, i - 1))
            start = i
        elif not d and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(down) - 1))
    dur_h = np.array([(b - a + 1) * spacing for a, b in runs])
    long_runs = [(w.t.iloc[a], (b - a + 1) * spacing) for a, b in runs if (b - a + 1) * spacing > 24.0]
    add("(b) contiguous non-operating runs", len(runs), "-")
    add("(b) longest run", dur_h.max() / 24.0, "days", f"starting {w.t.iloc[runs[int(dur_h.argmax())][0]]:%Y-%m-%d}")
    add("(b) runs longer than 24 h", len(long_runs), "-",
        "; ".join(f"{s:%Y-%m-%d %H:%M} ({h / 24:.1f} d)" for s, h in long_runs))
    add("(b) share of down-time in runs longer than 24 h", dur_h[dur_h > 24.0].sum() / dur_h.sum(), "-")
    add("(b) median run length", float(np.median(dur_h)), "h")
    a_alt = 1.0 - dur_h.sum() / 8760.0
    add("(b) annual availability if these outages are the year's only ones", a_alt, "-",
        f"the model uses {GPS_AVAILABILITY:.3f}, the window's own figure")
    for basis in BASES:
        g = _m6(basis)
        for arch in ("Direct-MED", "orc_med (path A)"):
            x = g[g.architecture == arch]
            r = x.loc[x.lcow_aud_per_m3.idxmin()]

            def at(k):
                return levelised_cost_of_water(
                    {"water": r.capex_water_aud, "heat_recovery": r.capex_heat_recovery_aud},
                    r.water_m3_per_year * k, r.net_electricity_kw, r.operating_hours * k,
                    real_discount_rate=r.real_discount_rate)["lcow_aud_per_m3"]

            _close(at(1.0), r.lcow_aud_per_m3, rel=1e-9, what=f"CR9 {arch} {basis}")
            add(f"(b) {arch} LCOW at that availability, {basis}", at(a_alt / GPS_AVAILABILITY), "A$/m3",
                f"model value {r.lcow_aud_per_m3:.3f} at {GPS_AVAILABILITY:.3f}")
            if basis == "bhojwani_global":
                note(f"      {arch}: A${r.lcow_aud_per_m3:.3f} at {GPS_AVAILABILITY:.3f} availability -> "
                     f"A${at(a_alt / GPS_AVAILABILITY):.3f} at {a_alt:.3f}")
    note(f"  (b) {int(down.sum())} non-operating records in {len(runs)} runs; longest "
         f"{dur_h.max() / 24:.1f} days; {len(long_runs)} runs > 24 h hold "
         f"{dur_h[dur_h > 24.0].sum() / dur_h.sum():.0%} of the down-time")

    adp = m1["calculated_adp_c"]
    for k, v in (("min", adp.min()), ("mean", adp.mean()), ("max", adp.max()), ("std", adp.std(ddof=1))):
        add(f"(c) acid dew point - {k}", v, "C", "per row, Module 1")
    add("(c) acid dew point - correlation with load", np.corrcoef(adp, m1["Unit Load (MW)"])[0, 1], "-")
    for t in (85, 90, 95, 100):
        add(f"(c) mean corrosion index, {t} C track", m1[f"corrosion_index_{t}"].mean(), "-",
            "index, not a rate - argue from the ADP margin")
    note(f"  (c) ADP {adp.min():.1f}-{adp.max():.1f} C (mean {adp.mean():.1f}, sd {adp.std(ddof=1):.2f}); "
         f"corr with load {np.corrcoef(adp, m1['Unit Load (MW)'])[0, 1]:+.2f}")

    sw_all = w["Sea water intake temp"]
    sw_op = pd.read_csv(OUT / "gps_module3_output_direct_med_85.csv")["Sea water intake temp"]
    for lab, s in (("all records", sw_all), ("operating rows", sw_op)):
        for k, v in (("min", s.min()), ("mean", s.mean()), ("max", s.max()), ("std", s.std(ddof=1))):
            add(f"(d) seawater intake, {lab} - {k}", v, "C", f"n = {len(s)}")
    d = m6._load_design_row_cached(str(DATA))
    g = pd.read_csv(DATA / "GPS_Operating_Data_Clean.csv"); g.columns = g.columns.str.strip()
    g["k"] = g["Timestamp"].astype(str).str.strip()
    sea = w.assign(k=w["Timestamp"].astype(str).str.strip())[["k", "Sea water intake temp"]]
    dfm = pd.merge(g, sea, on="k", how="inner")
    add("(d) seawater on the ORC design row (maximum gas flow)", dfm.loc[d["idx"], "Sea water intake temp"],
        "C", f"{dfm.loc[d['idx'], 'Timestamp']}")
    add("(d) RO sizing seawater (median of operating rows)",
        float(np.percentile(m6._seawater_rows(str(DATA))[0], m6.SEAWATER_DESIGN_PERCENTILE)), "C",
        "SEAWATER_DESIGN_PERCENTILE = 50")
    monthly = w.groupby(w.t.dt.to_period("M"))["Sea water intake temp"].agg(["mean", "count"])
    for per, r in monthly.iterrows():
        add(f"(d) monthly mean seawater, {per}", r["mean"], "C", f"{int(r['count'])} records")
    days = (w.t - w.t.iloc[0]).dt.total_seconds().values / 86400.0
    trend = np.polyfit(days, sw_all.values, 1)[0] * 30.44
    add("(d) linear trend across the window", trend, "C per month")
    fit = _seawater_fit()
    add("(d) full-year sinusoid - mean", fit["mean"], "C", "fitted to five months; crude (CR12)")
    add("(d) full-year sinusoid - amplitude", fit["amplitude"], "C")
    add("(d) full-year sinusoid - warmest day of year", fit["peak_day"], "day")
    add("(d) full-year sinusoid - annual minimum", fit["mean"] - fit["amplitude"], "C")
    add("(d) full-year sinusoid - annual maximum", fit["mean"] + fit["amplitude"], "C")
    add("(d) full-year sinusoid - R squared / RMSE", fit["r2"], "-", f"RMSE {fit['rmse']:.2f} C")
    note(f"  (d) seawater {sw_op.min():.1f}-{sw_op.max():.1f} C on operating rows; monthly means "
         + ", ".join(f"{p.strftime('%b')} {r['mean']:.1f}" for p, r in monthly.iterrows())
         + f"; trend {trend:+.2f} C/month")
    note(f"      full-year sinusoid: {fit['mean']:.1f} +/- {fit['amplitude']:.1f} C, warmest on day "
         f"{fit['peak_day']:.0f}, R2 {fit['r2']:.3f}")
    # (e) water dew point - where water itself condenses on the tubes
    m1s = m1.copy()
    m1s.columns = m1s.columns.str.strip()
    y = m1s["flue_gas_y_h2o"].astype(float)
    y = y / 100.0 if y.max() > 1.0 else y
    p_tot = P_AMBIENT_KPA + (m1s["Fabric Filter Outlet Pressure (kPa) A side"]
                             + m1s["Fabric Filter Outlet Pressure (kPa) B side"]) / 2.0
    t_dew = np.array([CP.PropsSI("T", "P", yy * pp * 1000.0, "Q", 0, "Water") - 273.15
                      for yy, pp in zip(y, p_tot)])
    for k, v in (("min", t_dew.min()), ("mean", t_dew.mean()), ("max", t_dew.max())):
        add(f"(e) flue-gas water dew point - {k}", v, "C", "H2O partial pressure at the fabric-filter outlet")
    g6 = _m6("bhojwani_global")
    lrs = {}
    for fam, test in FAMILIES:
        x6 = g6[g6.architecture.map(test)]
        if fam != "Conventional SWRO" and len(x6):
            lrs[fam] = float(x6.loc[x6.lcow_aud_per_m3.idxmin()].loop_return_c)
            add(f"(e) loop return, {fam}", lrs[fam], "C", "the cold-end tube wall sits ~2 K above it")
    note(f"  (e) water dew point {t_dew.min():.1f}-{t_dew.max():.1f} C (mean {t_dew.mean():.1f}); loop returns "
         + ", ".join(f"{k} {v:.1f}" for k, v in lrs.items()) + " C")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "operating_data_characterisation.csv", index=False)
    return df


# ==========================================================================
# CR10 - price of retreating from the corrosion peak
# ==========================================================================
_SEC_AMB = {}


def _conventional_swro_lcow(water, anchor):
    """Conventional SWRO at a given capacity - module6_lcow.run's no-heat reference,
    re-priced so that a penalty is always taken at matched capacity."""
    if not _SEC_AMB:
        _SEC_AMB["v"] = float(np.mean([_sec(t) for t in m6._seawater_rows(str(DATA))[0]]))
    return m6.assemble_architecture(
        name="Conventional SWRO (no heat recovery)", technology="SWRO", design_heat_kw=0.0,
        water_m3_per_day=float(water), hx_area_m2=0.0, fan_kw=0.0, loop_pump_kw=0.0,
        orc_gross_kw=0.0, process_sec_kwh_m3=_SEC_AMB["v"], anchor=anchor)["lcow_aud_per_m3"]


def cr10():
    note("== CR10 LCOW of retreating from the corrosion peak (85 -> 100 C)")
    m1 = pd.read_csv(OUT / "GPS_Module1_Output.csv")
    adp = m1["calculated_adp_c"].mean()
    opt = _opt()
    recs = []
    for basis in BASES:
        m = _m6(basis)
        ref = m[m.architecture.str.startswith("Conventional SWRO")].iloc[0]
        _close(_conventional_swro_lcow(ref.water_m3_per_day, basis), ref.lcow_aud_per_m3,
               what=f"CR10 conventional SWRO {basis}")
        for _, r in opt[opt.architecture == "orc_ro_current"].iterrows():
            rec = {"cost_basis": basis, "architecture": "ORC+RO", "path": r.path,
                   "track_c": r.track_c, "fan_feasible": bool(r.fan_feasible)}
            name = f"orc_ro_current (path {r.path})"
            hit = m[(m.architecture == name) & (m.track_c == r.track_c)]
            rec["selected_by_module6"] = not hit.empty
            if r.fan_feasible:
                res, perf = orc_ro_lcow(r, basis)
                rec.update(lcow_aud_per_m3=res["lcow_aud_per_m3"], water_m3_per_day=perf["water_m3_per_day"])
                if not hit.empty:
                    _close(res["lcow_aud_per_m3"], hit.lcow_aud_per_m3.iloc[0], what=f"CR10 {name} {r.track_c}")
            recs.append(rec)
        for _, r in m[m.architecture.str.startswith("Direct-RO")].iterrows():
            recs.append({"cost_basis": basis, "architecture": "Direct-RO", "path": "-",
                         "track_c": r.track_c, "fan_feasible": True, "selected_by_module6": True,
                         "lcow_aud_per_m3": r.lcow_aud_per_m3, "water_m3_per_day": r.water_m3_per_day})
        for rec in recs:
            if rec["cost_basis"] == basis and "lcow_aud_per_m3" in rec:
                same = _conventional_swro_lcow(rec["water_m3_per_day"], basis)
                rec["conventional_swro_same_capacity_aud_per_m3"] = same
                rec["penalty_vs_conventional_swro_aud_per_m3"] = rec["lcow_aud_per_m3"] - same
    df = pd.DataFrame(recs)
    ref = df[df.track_c == 85.0].set_index(["cost_basis", "architecture", "path"]).lcow_aud_per_m3
    key = list(zip(df.cost_basis, df.architecture, df.path))
    df["delta_vs_85c_aud_per_m3"] = [lc - ref.get(k, np.nan) for lc, k in zip(df.lcow_aud_per_m3, key)]
    df["delta_vs_85c_pct"] = [100.0 * d / ref.get(k, np.nan) for d, k in zip(df.delta_vs_85c_aud_per_m3, key)]
    df["mean_corrosion_index"] = [m1[f"corrosion_index_{int(t)}"].mean() for t in df.track_c]
    df["max_corrosion_index"] = [m1[f"corrosion_index_{int(t)}"].max() for t in df.track_c]
    df["gas_exit_minus_mean_adp_c"] = df.track_c - adp
    df = df.sort_values(["cost_basis", "architecture", "path", "track_c"]).reset_index(drop=True)
    df.to_csv(OUT / "corrosion_retreat_penalty.csv", index=False)
    for basis in BASES:
        for arch, path in (("ORC+RO", "A"), ("ORC+RO", "B"), ("Direct-RO", "-")):
            s = df[(df.cost_basis == basis) & (df.architecture == arch) & (df.path == path)
                   & df.lcow_aud_per_m3.notna()].set_index("track_c")
            if 85.0 in s.index and 100.0 in s.index:
                note(f"  {basis:15s} {arch:9s} {path}: A${s.lcow_aud_per_m3[85.0]:.3f} at 85 C -> "
                     f"A${s.lcow_aud_per_m3[100.0]:.3f} at 100 C "
                     f"({s.delta_vs_85c_aud_per_m3[100.0]:+.3f}, {s.delta_vs_85c_pct[100.0]:+.1f} %)")
    for basis in BASES:
        s = df[(df.cost_basis == basis) & (df.architecture == "Direct-RO")].set_index("track_c")
        note(f"  {basis:15s} heat-recovery penalty at matched capacity: "
             + ", ".join(f"{int(t)} C {v:.3f}" for t, v in s.penalty_vs_conventional_swro_aud_per_m3.items()))
    note(f"  mean ADP {adp:.1f} C: the gas leaves {85 - adp:+.1f} K from it at 85 C, {100 - adp:+.1f} K at 100 C")
    return df


# ==========================================================================
# CR15 - is any subcritical ORC worth building, at any track?
# ==========================================================================
CR15_TRACKS = (85.0, 90.0, 95.0, 100.0)
# The lowest built-in ratio of any ORC expander type: scroll, volume ratio
# 1.5-5 (Alshammari, Usman & Pesyridis 2018). A generous bar for "a real
# expander"; the verdict below does not depend on it.
CR15_MIN_EXPANDER_PR = 1.5
_TRACK_CTX = {}


def _track_point(job):
    """One ORC+RO design at (path, track, loop return), set up exactly as run_case does."""
    path, track, lr = job
    from loop_optimiser import _solver_a, _solver_b
    if track not in _TRACK_CTX:
        _TRACK_CTX[track] = _ctx_track(track)
    c = _TRACK_CTX[track]
    solver = partial(_solver_a if path == "A" else _solver_b, q_in=c["q_in"], tc=TC_C, t_hot=c["t_hot"])
    e = evaluate_loop_return(float(lr), lambda hh, hr, md: solver(hh, hr, md),
                             c["t_hot"], c["q_target"], c["design"]["t_in_c"], c["c_gas"],
                             c["gt"], c["n_tubes"], DUCT_HEIGHT_M)
    row = {"path": path, "track_c": track, "loop_return_c": lr, "feasible": e is not None}
    if e is None:
        return row
    entry = {"orc_evap_ua_w_per_k": area_from_pinch_profile_m2(e["pinch"]["profile"], e["pinch"]["q_total_w"], 1.0),
             "tc_c": TC_C, "q_cond_kw": e["q_cond_kw"], "area_total_m2": e["area_total_m2"],
             "fan_kw": e["fan_kw"], "loop_pump_kw": e["loop_pump_kw"], "w_net_kw": e["w_net_kw"]}
    rec, perf = orc_ro_lcow(entry, "bhojwani_global")
    row.update(rows=e["rows"], min_pinch_k=e["pinch"]["min_pinch_k"], orc_t_evap_c=e["operating_point_c"],
               w_net_kw=e["w_net_kw"], orc_heat_in_kw=c["q_in"], fan_kw=e["fan_kw"],
               loop_pump_kw=e["loop_pump_kw"], area_total_m2=e["area_total_m2"])
    if rec is not None:
        row.update(lcow_aud_per_m3=rec["lcow_aud_per_m3"],
                   electricity_component_aud_per_m3=rec["electricity_component_aud_per_m3"],
                   heat_side_kw=rec["heat_side_kw"], water_m3_per_day=perf["water_m3_per_day"])
    return row


def cr15(pool):
    """Path A across its whole loop-return range at every track, against Direct-RO at the
    same track."""
    note("== CR15 subcritical ORC (path A) at every track: is any worth building?")
    from path_b_transcritical import ORC_TRANS_FLUID
    cap = m6._MED_ROW_CAP
    m = _m6("bhojwani_global")
    design = {t: float(_design_entry("A", track=t).loop_return_c) for t in CR15_TRACKS}
    jobs = []
    for t in CR15_TRACKS:
        upper = min(t - LOSS_BUFFER_DELTA_T - 1.0, t - 5.0)  # the optimiser's own bounds, as CR5
        jobs += [("A", t, float(lr)) for lr in np.round(np.arange(TC_C + 2.0, upper + 1e-9, 0.5), 3)]
        jobs.append(("A", t, design[t]))
    jobs.append(("B", 90.0, float(_design_entry("B", track=90.0).loop_return_c)))
    df = pd.DataFrame(pool.map(_track_point, jobs))

    # reproduction: the optimiser's path A design at every track, and Module 6's LCOW
    # at two tracks and both paths (path A is reported only at 85 C, path B from 90 C)
    for t in CR15_TRACKS:
        r = df[(df.path == "A") & (df.track_c == t) & np.isclose(df.loop_return_c, design[t])].iloc[0]
        o = _design_entry("A", track=t)
        _close(r.area_total_m2, o.area_total_m2, what=f"CR15 path A area at {t:.0f} C")
        _close(r.rows, o.rows, what=f"CR15 path A rows at {t:.0f} C")
    for path, t in (("A", 85.0), ("B", 90.0)):
        lr = design[t] if path == "A" else float(_design_entry("B", track=t).loop_return_c)
        r = df[(df.path == path) & (df.track_c == t) & np.isclose(df.loop_return_c, lr)].iloc[0]
        ref = m[(m.architecture == f"orc_ro_current (path {path})") & (m.track_c == t)].lcow_aud_per_m3.iloc[0]
        _close(r.lcow_aud_per_m3, ref, what=f"CR15 path {path} LCOW at {t:.0f} C")
    note("  reproduces the optimiser's path A design at all four tracks, and Module 6's LCOW for"
         " path A at 85 C and path B at 90 C")

    df = df[df.path == "A"].copy()
    df["is_model_design"] = [bool(np.isclose(lr, design[t])) for t, lr in zip(df.track_c, df.loop_return_c)]
    df = df.sort_values(["track_c", "loop_return_c"]).reset_index(drop=True)
    df["fan_feasible"] = df["feasible"] & (df["rows"].fillna(1e9) <= cap)
    p_cond = CP.PropsSI("P", "T", TC_C + 273.15, "Q", 0, ORC_TRANS_FLUID)
    df["pressure_ratio"] = [CP.PropsSI("P", "T", te + 273.15, "Q", 0, ORC_TRANS_FLUID) / p_cond
                            if ok else np.nan for te, ok in zip(df.orc_t_evap_c, df.feasible)]
    df["orc_efficiency_pct"] = 100.0 * df.w_net_kw / df.orc_heat_in_kw
    d = m[m.architecture == "Direct-RO (no ORC)"].set_index("track_c")
    df["direct_ro_lcow_aud_per_m3"] = df.track_c.map(d.lcow_aud_per_m3)
    df["lcow_vs_direct_ro_aud_per_m3"] = df.lcow_aud_per_m3 - df.direct_ro_lcow_aud_per_m3
    # the ORC's own effect on the heat side's power: what it makes, less the extra
    # fan and loop-pump power its warmer loop return costs against Direct-RO. Both
    # are annual means over the operating records, as the LCOW charges them
    df["net_power_vs_direct_ro_kw"] = df.heat_side_kw - df.track_c.map(d.heat_side_kw)
    # LCOW is linear in the electricity price (common/lcow.py), so the price at which a
    # design would match Direct-RO exists only if it imports less electricity per m3
    imp_o = df.electricity_component_aud_per_m3 / ELECTRICITY_AUD_PER_MWH
    imp_d = df.track_c.map(d.electricity_component_aud_per_m3) / ELECTRICITY_AUD_PER_MWH
    rest_o = df.lcow_aud_per_m3 - df.electricity_component_aud_per_m3
    rest_d = df.track_c.map(d.lcow_aud_per_m3 - d.electricity_component_aud_per_m3)
    df["break_even_electricity_aud_per_mwh"] = np.where(imp_o < imp_d, (rest_o - rest_d) / (imp_d - imp_o), np.nan)

    for t in CR15_TRACKS:
        s = df[(df.track_c == t) & df.fan_feasible]
        f = df[(df.track_c == t) & df.feasible]
        fl = df[(df.track_c == t) & df.is_model_design].iloc[0]
        top = s.loc[s.pressure_ratio.idxmax()]
        big = f.loc[f.pressure_ratio.idxmax()]
        falls = int((np.diff(s.lcow_aud_per_m3.values) < -1e-9).sum())
        note(f"  {t:.0f} C track: fits the fan from {s.loop_return_c.min():.2f} to {s.loop_return_c.max():.2f} C "
             f"of loop return ({int(s.rows.min())}-{int(s.rows.max())} rows of {cap})")
        for lab, r in (("floor (model)", fl), ("highest PR in fan", top), ("ignoring the fan", big)):
            note(f"    {lab:18s} return {r.loop_return_c:5.2f} C, {int(r.rows):2d} rows: boils {r.orc_t_evap_c:5.1f} C, "
                 f"PR {r.pressure_ratio:.2f}, {r.w_net_kw:4.0f} kW ({r.orc_efficiency_pct:.2f} %), "
                 f"A${r.lcow_aud_per_m3:.3f} = Direct-RO {r.lcow_vs_direct_ro_aud_per_m3:+.3f}, "
                 f"power vs Direct-RO {r.net_power_vs_direct_ro_kw:+.0f} kW")
        note(f"    within the fan: LCOW falls as loop return rises at {falls} of {len(s) - 1} steps; "
             f"{int((s.lcow_vs_direct_ro_aud_per_m3 < 0).sum())} of {len(s)} beat Direct-RO; "
             f"{int((s.net_power_vs_direct_ro_kw > 0).sum())} make more power than Direct-RO")
    ok = df[df.fan_feasible & (df.pressure_ratio >= CR15_MIN_EXPANDER_PR) & (df.lcow_vs_direct_ro_aud_per_m3 < 0)]
    be = df[df.feasible].break_even_electricity_aud_per_mwh.dropna()
    note(f"  designs that fit the fan, reach PR >= {CR15_MIN_EXPANDER_PR} and beat Direct-RO: {len(ok)} of "
         f"{int(df.fan_feasible.sum())} fan-feasible ({int(df.feasible.sum())} feasible)")
    note(f"  cheapest design against Direct-RO, ignoring the fan: "
         f"{df[df.feasible].lcow_vs_direct_ro_aud_per_m3.min():+.3f} A$/m3; designs with a break-even "
         f"electricity price: {len(be)}" + (f", lowest A${be.min():,.0f}/MWh" if len(be) else ""))
    cols = ["track_c", "loop_return_c", "feasible", "fan_feasible", "is_model_design", "rows",
            "min_pinch_k", "orc_t_evap_c", "pressure_ratio", "w_net_kw", "orc_heat_in_kw",
            "orc_efficiency_pct", "fan_kw", "loop_pump_kw", "heat_side_kw", "area_total_m2", "water_m3_per_day",
            "lcow_aud_per_m3", "direct_ro_lcow_aud_per_m3", "lcow_vs_direct_ro_aud_per_m3",
            "net_power_vs_direct_ro_kw", "break_even_electricity_aud_per_mwh"]
    df[cols].to_csv(OUT / "subcritical_orc_all_tracks.csv", index=False)
    return df


# ==========================================================================
# CR12 - seawater temperature, 20-30 C and annualised
# ==========================================================================
def cr12():
    note("== CR12 seawater temperature (ORC+RO design, global basis)")
    entry = _design_entry("A")
    base, perf = orc_ro_lcow(entry, "bhojwani_global")
    m6_row = _m6("bhojwani_global").query(
        "architecture == 'orc_ro_current (path A)' and track_c == @TRACK_C").iloc[0]
    _close(base["lcow_aud_per_m3"], m6_row.lcow_aud_per_m3, what="CR12 design point")
    q = float(entry.q_cond_kw)
    t_d, feed_d, water_d = perf["design_seawater_c"], perf["feed_kg_s"], perf["water_m3_per_day"]
    recs = []
    for interp in ("capacity_tracks", "capacity_fixed"):
        for t in sorted({20.0, 22.0, 24.0, 26.0, 28.0, 30.0, round(t_d, 3)}):
            if interp == "capacity_tracks":
                water = q / (SEAWATER_CP_KJ_KGK * (CEILING_C - t)) * RO_RECOVERY * 86.4
                t_feed, sink = CEILING_C, t
            else:
                water = water_d
                t_feed, sink = min(t + q / (feed_d * SEAWATER_CP_KJ_KGK), CEILING_C), t_d
            rec = _assemble_orc_ro(entry, "bhojwani_global", water, _sec(t_feed), _sec(t), sink)
            recs.append({"t_seawater_c": t, "interpretation": interp, "is_design": abs(t - t_d) < 1e-3,
                         "water_m3_per_day": water, "t_feed_preheated_c": t_feed,
                         "sec_kwh_m3": rec["sec_effective_kwh_m3"],
                         "lcow_aud_per_m3": rec["lcow_aud_per_m3"]})
    df = pd.DataFrame(recs)
    ref = df[df.is_design].set_index("interpretation").lcow_aud_per_m3
    df["delta_vs_design_aud_per_m3"] = df.lcow_aud_per_m3 - df.interpretation.map(ref)

    # Annualised, on the model's own interpretation (capacity fixed). Each
    # operating row keeps its preheat rise at the built feed rate; the seawater
    # it starts from follows the fitted year instead of the observed five months.
    t_rows, gas = m6._seawater_rows(str(DATA))
    grid = np.arange(10.0, 40.0001, 0.01)
    sec_grid = np.array([_sec(t) for t in grid])
    S = lambda t: np.interp(t, grid, sec_grid)
    fit = _seawater_fit()
    hours = annual_operating_hours("ORC_RO")
    w_pre, w_amb = hours["preheated_h"] / hours["total_h"], hours["ambient_h"] / hours["total_h"]

    def shift(q_kw, feed):
        rise = q_kw * (gas / gas.max()) / (feed * SEAWATER_CP_KJ_KGK)
        pre = lambda temps: S(np.minimum(temps[:, None] + rise[None, :], CEILING_C)).mean()
        return pre(fit["year"]) - pre(fit["window"]), S(fit["year"]).mean() - S(fit["window"]).mean()

    _close(S(np.minimum(t_rows + q * (gas / gas.max()) / (feed_d * SEAWATER_CP_KJ_KGK), CEILING_C)).mean(),
           perf["sec_preheated_kwh_m3"], rel=1e-4, what="CR12 SEC interpolant")
    d_pre, d_amb = shift(q, feed_d)
    ann = _assemble_orc_ro(entry, "bhojwani_global", water_d, perf["sec_preheated_kwh_m3"] + d_pre,
                           perf["sec_ambient_kwh_m3"] + d_amb, t_d)
    price = ELECTRICITY_AUD_PER_MWH / 1000.0
    _close(ann["lcow_aud_per_m3"] - base["lcow_aud_per_m3"], price * (d_pre * w_pre + d_amb * w_amb),
           rel=1e-6, what="CR12 closed form")
    dro = _m6("bhojwani_global").query("architecture == 'Direct-RO (no ORC)' and track_c == @TRACK_C").iloc[0]
    perf_dro = m6.ro_annual_performance(float(dro.design_heat_kw), CEILING_C, str(DATA))
    d_pre_dro, _ = shift(float(dro.design_heat_kw), perf_dro["feed_kg_s"])
    dl = {"ORC+RO": price * (d_pre * w_pre + d_amb * w_amb),
          "Direct-RO": price * (d_pre_dro * w_pre + d_amb * w_amb),
          "Conventional SWRO": price * d_amb}
    extra = pd.DataFrame([{"t_seawater_c": float(fit["year"].mean()), "interpretation": "annualised_fit",
                           "is_design": False, "water_m3_per_day": water_d, "t_feed_preheated_c": np.nan,
                           "sec_kwh_m3": ann["sec_effective_kwh_m3"],
                           "lcow_aud_per_m3": ann["lcow_aud_per_m3"],
                           "delta_vs_design_aud_per_m3": ann["lcow_aud_per_m3"] - base["lcow_aud_per_m3"]}])
    df = pd.concat([df, extra], ignore_index=True)
    df.to_csv(OUT / "seawater_sensitivity.csv", index=False)
    for interp in ("capacity_tracks", "capacity_fixed"):
        s = df[df.interpretation == interp].set_index("t_seawater_c")
        note(f"  {interp:15s}: 20 C -> A${s.lcow_aud_per_m3[20.0]:.3f} "
             f"({s.water_m3_per_day[20.0]:,.0f} m3/d); 30 C -> A${s.lcow_aud_per_m3[30.0]:.3f} "
             f"({s.water_m3_per_day[30.0]:,.0f} m3/d); design {t_d:.2f} C")
    note(f"  the model uses capacity_fixed (feed sized at the median intake)")
    note(f"  annualised (fitted year {fit['year'].min():.1f}-{fit['year'].max():.1f} C vs window "
         f"{fit['window'].min():.1f}-{fit['window'].max():.1f} C): SEC preheated {d_pre:+.4f}, "
         f"ambient {d_amb:+.4f} kWh/m3")
    note(f"    LCOW shift ORC+RO {dl['ORC+RO']:+.4f}, Direct-RO {dl['Direct-RO']:+.4f}, "
         f"conventional SWRO {dl['Conventional SWRO']:+.4f} A$/m3 -> heat-recovery penalty moves "
         f"{dl['Direct-RO'] - dl['Conventional SWRO']:+.4f} A$/m3")
    return df, dl


# ==========================================================================
# CR4 - asset life by pool
# ==========================================================================
FAMILIES = (("Conventional SWRO", lambda a: a.startswith("Conventional SWRO")),
            ("Direct-RO", lambda a: a.startswith("Direct-RO")),
            ("ORC+RO", lambda a: a.startswith("orc_ro_current")),
            ("Direct-MED", lambda a: a == "Direct-MED"),
            ("ORC+MED", lambda a: a.startswith("orc_med")))
STRANDS_WITH_STATION = {"Direct-MED", "ORC+MED"}   # MED is useless without the heat


def cr4():
    note("== CR4 asset life by pool (corrosion and closure cases)")
    g = _m6("bhojwani_global")
    picks = {}
    for fam, test in FAMILIES:
        x = g[g.architecture.map(test)]
        picks[fam] = x.loc[x.lcow_aud_per_m3.idxmin()]
    recs = []
    for basis in BASES:
        m = _m6(basis)
        for fam, p in picks.items():
            r = m[(m.architecture == p.architecture)
                  & ((m.track_c == p.track_c) | (m.track_c.isna() & pd.isna(p.track_c)))].iloc[0]

            def lcow(lw, lh):
                return levelised_cost_of_water(
                    {"water": r.capex_water_aud, "heat_recovery": r.capex_heat_recovery_aud},
                    r.water_m3_per_year, r.net_electricity_kw, r.operating_hours,
                    real_discount_rate=r.real_discount_rate, life_water_years=lw, life_heat_years=lh)

            b = lcow(ASSET_LIFE_WATER_YEARS, ASSET_LIFE_HEAT_RECOVERY_YEARS)
            _close(b["lcow_aud_per_m3"], r.lcow_aud_per_m3, rel=1e-9, what=f"CR4 {fam} {basis}")
            cases = [("base", ASSET_LIFE_HEAT_RECOVERY_YEARS, ASSET_LIFE_WATER_YEARS)]
            for n in (20.0, 15.0, 10.0):
                cases.append(("corrosion", n, ASSET_LIFE_WATER_YEARS))
                cases.append(("closure", n, n if fam in STRANDS_WITH_STATION else ASSET_LIFE_WATER_YEARS))
            for case, lh, lw in cases:
                res = lcow(lw, lh)
                recs.append({"cost_basis": basis, "case": case, "architecture": fam,
                             "design": f"{r.architecture} @ {'' if pd.isna(r.track_c) else int(r.track_c)}",
                             "heat_life_yr": lh, "water_life_yr": lw,
                             "annual_capital_charge_aud": res["annual_capital_charge_aud"],
                             "lcow_aud_per_m3": res["lcow_aud_per_m3"],
                             "delta_vs_base_aud_per_m3": res["lcow_aud_per_m3"] - b["lcow_aud_per_m3"]})
    df = pd.DataFrame(recs)
    df["rank"] = df.groupby(["cost_basis", "case", "heat_life_yr"]).lcow_aud_per_m3.rank().astype(int)
    df.to_csv(OUT / "asset_life_sensitivity.csv", index=False)
    for basis in BASES:
        d = df[df.cost_basis == basis]
        for case in ("corrosion", "closure"):
            s = d[(d.case == case) & (d.heat_life_yr == 10.0)].set_index("architecture")
            note(f"  {basis:15s} {case:9s} 10-yr: " + ", ".join(
                f"{a} {s.lcow_aud_per_m3[a]:.2f} ({s.delta_vs_base_aud_per_m3[a]:+.2f})"
                for a in ("Conventional SWRO", "Direct-RO", "ORC+RO", "Direct-MED", "ORC+MED")))
        pen = d.pivot_table(index=["case", "heat_life_yr"], columns="architecture", values="lcow_aud_per_m3")
        pen = (pen["Direct-RO"] - pen["Conventional SWRO"])
        note(f"  {basis:15s} heat-recovery penalty: base {pen[('base', 25.0)]:.3f} -> "
             f"{pen[('corrosion', 10.0)]:.3f} at a 10-yr heat-pool life; order unchanged in every case: "
             f"{bool((d.groupby(['case', 'heat_life_yr']).apply(lambda x: list(x.sort_values('lcow_aud_per_m3').architecture)).map(tuple).nunique() == 1))}")
    return df


def main():
    with mp.Pool(6) as pool:
        cr5(pool)
        cr15(pool)
    cr6()
    cr6_resolution()
    cr7()
    cr9()
    cr10()
    cr12()
    cr4()
    med_sizing_check()
    (OUT / "supporting_calculations_summary.txt").write_text("\n".join(SUMMARY) + "\n")
    print(f"\nwrote {OUT / 'supporting_calculations_summary.txt'}")


if __name__ == "__main__":
    main()
