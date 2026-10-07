"""
The loop-return optimisation: for every duct case, ORC+RO and ORC+MED
architecture (design_cases.ARCHITECTURES), extraction track and ORC path
(A subcritical, B transcritical), finds the LCOW-optimal loop return - the
minimum feasible value (common.loop_optimisation.find_minimum_feasible_loop_return)
- at the design record, and records the resulting exchanger, fan, pump and
ORC design together with its ID fan feasibility.

The (duct x architecture) cases are independent, so they run in a process
pool. The loop-return search starts at Tc + 2 K: below roughly Tc plus the
condenser pinch (the cold-end constraint) no loop return can be feasible.

Writes outputs/loop_optimisation_summary_lcow.csv. The slowest step of the
pipeline; run_all.py --skip-optimiser reuses its output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import multiprocessing as mp
from functools import partial

import pandas as pd

from common.costing import area_from_pinch_profile_m2
from common.fan_margin import row_caps_for_ducts

from common.gas_properties import dynamic_flue_gas_cp
from common.hx_loop_physics import duct_geometry, gas_side_heat_transfer
from common.loop_optimisation import optimise_loop_return, find_minimum_feasible_loop_return
from common.constants import (
    N_GAS_TRAINS, N_TRAINS_FITTED,
    LOSS_BUFFER_DELTA_T, HX_SHELL_EFFICIENCY, TARGET_TEMPERATURES,
    ORC_EVAPORATOR_PINCH_C, PITCH_RATIO, DUCT_HEIGHT_M,
)
from path_a_subcritical import solve_path_a
from path_b_transcritical import solve_path_b, ORC_TRANS_FLUID
from design_cases import DUCT_CASES, ARCHITECTURES, load_design_row


def _solver_a(h_hot, h_ret, m_dot, q_in, tc, t_hot):
    return solve_path_a(t_evap_unconstrained_c=t_hot - ORC_EVAPORATOR_PINCH_C, t_cond_c=tc,
                         q_in_kw=q_in, h_loop_hot=h_hot, h_loop_return=h_ret,
                         m_dot_loop=m_dot, fluid=ORC_TRANS_FLUID)


def _solver_b(h_hot, h_ret, m_dot, q_in, tc, t_hot):
    return solve_path_b(t3_search_low_c=tc + 3.0, t3_search_high_c=t_hot - 2.0, t_cond_c=tc,
                         q_in_kw=q_in, h_loop_hot=h_hot, h_loop_return=h_ret,
                         m_dot_loop=m_dot, n_scan=40)


def run_case(case, design, max_dp_pa=None, objective="lcow", fan_row_caps=None):
    duct_label, duct_w, duct_h, arch, tc = case
    # Hard ID fan constraint: the deepest bank whose added shaft power
    # keeps EVERY operating row inside the 2450 kW motor rating. Designs
    # exceeding it are kept but flagged, not silently dropped - the
    # unconstrained optimum is itself a reportable quantity.
    row_cap = (fan_row_caps or {}).get(duct_label)

    n_tubes, a_flow_ref = duct_geometry(duct_w, PITCH_RATIO)
    a_flow = a_flow_ref * (duct_h / DUCT_HEIGHT_M)
    gas_transport = gas_side_heat_transfer(design["t_in_c"], design["m_dot_fg"],
                                            design["p_gauge"], design["dust"], a_flow)
    face_v = gas_transport["volumetric_flow_m3s_duct"] / (duct_w * duct_h)

    records = []
    for t_target in TARGET_TEMPERATURES:
        if design["t_in_c"] <= t_target:
            continue
        cp_d = dynamic_flue_gas_cp((design["t_in_c"] + t_target) / 2.0, design["mole_fractions"])
        q_target = (design["m_dot_fg"] * (N_TRAINS_FITTED / N_GAS_TRAINS) * cp_d * (design["t_in_c"] - t_target))
        q_in = q_target * HX_SHELL_EFFICIENCY
        c_gas = (design["m_dot_fg"] / N_GAS_TRAINS) * cp_d
        t_hot = t_target - LOSS_BUFFER_DELTA_T

        for path_label, fn in (("A", _solver_a), ("B", _solver_b)):
            solver = partial(fn, q_in=q_in, tc=tc, t_hot=t_hot)
            if objective == "lcow":
                # LCOW-optimal = minimum feasible loop return, valid because
                # LCOW rises monotonically with loop return
                best = find_minimum_feasible_loop_return(
                    path_solver=lambda hh, hr, md, _s=solver: _s(hh, hr, md),
                    t_water_hot_c=t_hot, q_target_kw=q_target, t_gas_in_c=design["t_in_c"],
                    c_gas_per_duct_kw_per_k=c_gas, gas_transport=gas_transport,
                    n_tubes_per_row=n_tubes, duct_height_m=duct_h,
                    lr_lower_bound_c=tc + 2.0, t_gas_out_c=t_target)
            else:
                out = optimise_loop_return(
                    path_solver=lambda hh, hr, md, _s=solver: _s(hh, hr, md),
                    t_water_hot_c=t_hot, q_target_kw=q_target, t_gas_in_c=design["t_in_c"],
                    t_gas_out_c=t_target, c_gas_per_duct_kw_per_k=c_gas,
                    gas_transport=gas_transport, n_tubes_per_row=n_tubes, duct_height_m=duct_h,
                    lr_min_c=tc + 2.0, max_dp_pa=max_dp_pa)
                best = out["best"]
            if best is None:
                continue
            records.append({
                "duct": duct_label, "duct_area_m2": duct_w * duct_h, "face_velocity_ms": face_v,
                "architecture": arch, "tc_c": tc, "track_c": t_target, "path": path_label,
                "loop_return_c": best["loop_return_c"],
                "operating_point_c": best["operating_point_c"],
                "m_dot_loop_kg_s": best["m_dot_loop_kg_s"],
                "area_total_m2": best["area_total_m2"], "rows": best["rows"],
                "dp_pa": best["dp_pa"], "fan_kw": best["fan_kw"],
                "loop_pump_kw": best["loop_pump_kw"], "w_net_kw": best["w_net_kw"],
                "q_cond_kw": best["q_cond_kw"],
                "net_electricity_kw": best["w_net_kw"] - best["fan_kw"] - best["loop_pump_kw"],
                "min_pinch_k": best["pinch"]["min_pinch_k"],
                "max_rows_fan": row_cap,
                "fan_feasible": (row_cap is None or best["rows"] <= row_cap),
                # UA integrated segment-wise over the composite curve, so Module 6
                # can price the evaporator at any U without re-solving the cycle
                # (area = UA / U). Stored as UA rather than an area because U is
                # carried as a band, not a point value.
                "orc_evap_ua_w_per_k": area_from_pinch_profile_m2(
                    best["pinch"]["profile"], best["pinch"]["q_total_w"], 1.0),
            })

    print(f"  done: {duct_label:28s} {arch:16s} -> {len(records)} feasible designs", flush=True)
    return records


def main(data_dir, output_dir, workers=6, max_dp_pa=None, objective="lcow"):
    design = load_design_row(data_dir)
    print(f"design row idx={design['idx']}  T_in={design['t_in_c']:.2f}C  "
          f"m_dot_fg={design['m_dot_fg']:.1f} kg/s", flush=True)

    cases = [(dl, dw, dh, arch, tc)
             for (dl, dw, dh) in DUCT_CASES
             for arch, tc in ARCHITECTURES.items()]
    caps = row_caps_for_ducts(data_dir, DUCT_CASES)
    print("ID fan row caps (PF 0.85, clean): "
          + ", ".join(f"{k}={v}" for k, v in caps.items()), flush=True)
    print(f"{len(cases)} (duct x architecture) cases on {workers} workers\n", flush=True)

    with mp.Pool(workers) as pool:
        results = pool.map(partial(run_case, design=design, max_dp_pa=max_dp_pa,
                                    objective=objective, fan_row_caps=caps), cases)

    df = pd.DataFrame([r for group in results for r in group])
    suffix = "lcow" if objective == "lcow" else "netelec"
    out_path = Path(output_dir) / f"loop_optimisation_summary_{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(df)} rows)", flush=True)
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    (root / "outputs").mkdir(exist_ok=True)
    main(str(root / "data"), str(root / "outputs"))
