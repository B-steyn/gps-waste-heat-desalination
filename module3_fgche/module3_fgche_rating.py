"""
Module 3: the flue-gas exchanger (FGCHE), designed once and rated at every
operating record.

The exchanger is DESIGNED once at the design record (the highest flue-gas
flow, the standard full-load rating point), which fixes its area and UA.
That fixed geometry is then RATED against every record's own flow
conditions by epsilon-NTU (common/ntu_rating.py), rather than assuming every
record reaches the target temperature. TARGET_TEMPERATURES is the target the
hardware is designed for; the per-record gas and water outlet temperatures
are rating outputs.

The loop return temperature is an INPUT. Raising it narrows the loop dT,
which raises the loop flow and flattens the loop's T-Q curve, letting the
ORC working fluid evaporate hotter - but it also shrinks the exchanger's
LMTD and so needs more area, more tube rows and more fan power. Choosing it
is common/loop_optimisation.py's job; pass it via `loop_return_c_override`.
When it is omitted, a minimised loop return is derived from a one-off ORC
design-point solve and a warning is printed, since that choice is not
optimised.

The sizing chain (loop_return -> m_dot -> UA -> area -> rows -> dP -> fan)
lives in common/fgche_sizing.py and is shared with the optimiser, so the
two cannot drift apart. run_all.py runs this module for the Direct-MED
exchanger at the chosen loop returns; its per-record heat sets MED's P90
sizing and capacity factor.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module5_orc"))

from common.coal_combustion import calculate_dynamic_adp, combustion_mole_balance, calculate_corrosion_index
from common.gas_properties import dynamic_flue_gas_cp
from common.hx_loop_physics import duct_geometry, gas_side_heat_transfer
# train counts from the TAP_LOCATION switch (common/constants.py)
from common.constants import N_GAS_TRAINS, N_TRAINS_FITTED
from common.fgche_sizing import size_fgche, gas_side_fan_power, loop_pump_power_kw
from common.ntu_rating import rate_fixed_hx
from common.fluid_properties import ThermoCore
from common.orc_cycle import solve_orc_cycle
from common.constants import (
    P_AMBIENT_KPA, T_AMBIENT_CELSIUS, DUCT_WIDTH_M, PITCH_RATIO,
    P_WATER_LOOP_PA, ARCHITECTURE_WATER_RETURN_C, TARGET_TEMPERATURES,
    LOSS_BUFFER_DELTA_T, HX_SHELL_EFFICIENCY, TUBE_DO_M, DUCT_HEIGHT_M,
    ORC_EVAPORATOR_PINCH_C, ORC_CONDENSER_PINCH_C, ORC_MED_TC_RANGE_C,
    ORC_RO_TC_CURRENT_MEMBRANE_C, ORC_RO_TC_ADVANCED_MEMBRANE_C,
)

# selector options for downstream loop constraints: "direct_med", "orc_med", "orc_ro"
active_architecture = "direct_med"

# only used when active_architecture == "orc_ro" - Module 5 runs both
# membrane cases, so Module 3's ORC-branch sizing needs to too (each
# gets its own Tc, hence its own loop return temperature and hardware)
ro_membrane_case = "current"

t_ambient_celsius = T_AMBIENT_CELSIUS

_RO_TC_BY_CASE = {"current": ORC_RO_TC_CURRENT_MEMBRANE_C, "advanced": ORC_RO_TC_ADVANCED_MEMBRANE_C}


def _legacy_minimised_loop_return(t_target, q_target_design):
    """
    Fallback loop return, used only when this module runs on its own;
    run_all.py always supplies the designed loop return. Direct-MED
    returns its architecture constant;
    the ORC branches return the ORC pump-outlet temperature plus the
    condenser pinch, i.e. the MINIMUM loop return the cold-end pinch
    allows.

    That minimum is close to the worst valid choice: it maximises the
    loop's dT, which minimises loop mass flow, which STEEPENS the loop's
    T-Q curve so it plunges through the working fluid's boiling plateau
    and forces the evaporation temperature down. Prefer
    common/loop_optimisation.py. Returns None if the ORC design point is
    infeasible.
    """
    if active_architecture == "direct_med":
        return ARCHITECTURE_WATER_RETURN_C["direct_med"]

    t_evap = (t_target - LOSS_BUFFER_DELTA_T) - ORC_EVAPORATOR_PINCH_C
    if active_architecture == "orc_ro":
        tc = _RO_TC_BY_CASE[ro_membrane_case]
    elif active_architecture == "orc_med":
        tc = ORC_MED_TC_RANGE_C[0]
    else:
        raise ValueError(f"unknown architecture: {active_architecture}")

    result = solve_orc_cycle(t_evap_c=t_evap, t_cond_c=tc, q_in_kw=q_target_design * HX_SHELL_EFFICIENCY)
    if result is None:
        return None
    return result["state_points"][2]["t_c"] + ORC_CONDENSER_PINCH_C


def execute_master_module3_pipeline(flue_gas_csv, seawater_csv, output_dir,
                                     loop_return_c_override=None):
    coal_moles = calculate_dynamic_adp()
    thermo = ThermoCore(T0_celsius=t_ambient_celsius, P0_kpa=P_AMBIENT_KPA)

    print("loading and preprocessing raw source log files...")
    df_gas = pd.read_csv(flue_gas_csv)
    df_sea = pd.read_csv(seawater_csv)

    df_gas.columns = df_gas.columns.str.strip()
    df_sea.columns = df_sea.columns.str.strip()

    df_gas['timestamp_clean'] = df_gas['Timestamp'].astype(str).str.strip()
    df_sea['timestamp_clean'] = df_sea['Timestamp'].astype(str).str.strip()

    df_master = pd.merge(df_gas, df_sea[['timestamp_clean', 'Sea water intake temp']], on='timestamp_clean',
                         how='inner')

    print(f"timestamp merge: {len(df_gas)} flue-gas rows, {len(df_sea)} seawater rows -> {len(df_master)} matched")
    if len(df_master) != len(df_gas):
        print(f"[warning]: {len(df_gas) - len(df_master)} flue-gas rows did not find a seawater timestamp match.")
    if len(df_master) == 0:
        print("\n[critical error]: timestamp merge returned zero synchronized records.")
        return

    n_tubes_per_row, a_flow_minimum = duct_geometry(DUCT_WIDTH_M, PITCH_RATIO)

    label = f"{active_architecture}" + (f"_{ro_membrane_case}" if active_architecture == "orc_ro" else "")
    print(f"executing multi-track pipeline for configuration: {label}...\n")

    row_combustion = []
    for idx, row in df_master.iterrows():
        o2_meas = (row['A side flue gas o2 %'] + row['B side flue gas o2 %']) / 2.0
        p_gauge = (row['Fabric Filter Outlet Pressure (kPa) A side'] +
                   row['Fabric Filter Outlet Pressure (kPa) B side']) / 2.0
        m_dot_fg_total = row['Total gas flow kg/s']

        if m_dot_fg_total <= 1.0 or o2_meas >= 21.0:
            row_combustion.append(None)
            continue
        row_combustion.append(combustion_mole_balance(o2_meas, coal_moles, p_gauge, P_AMBIENT_KPA))

    # design row: full-load convention (highest flue gas flow among valid rows)
    valid_mask = [c is not None for c in row_combustion]
    design_idx = df_master.loc[valid_mask, 'Total gas flow kg/s'].idxmax()
    design_row = df_master.loc[design_idx]
    design_precomp = row_combustion[design_idx]

    for t_target in TARGET_TEMPERATURES:
        # --- STEP 1: characterise the design row ---
        t_in_design = design_row['FF_Outlet_Temp_Avg_C']
        m_dot_fg_design = design_row['Total gas flow kg/s']
        p_gauge_design = (design_row['Fabric Filter Outlet Pressure (kPa) A side'] +
                          design_row['Fabric Filter Outlet Pressure (kPa) B side']) / 2.0
        dust_design = (design_row['A Dust Burden (mg/M3)'] + design_row['B Dust Burden (mg/M3)']) / 2.0

        if t_in_design <= t_target:
            print(f"=== target {t_target}°c: design row cannot reach this target (T_in={t_in_design:.1f}C) - skipping track ===\n")
            continue

        mole_fractions_design = {"y_co2": design_precomp["y_co2"], "y_h2o": design_precomp["y_h2o"],
                                  "y_o2": design_precomp["y_o2"], "y_n2": design_precomp["y_n2"]}
        cp_design = dynamic_flue_gas_cp((t_in_design + t_target) / 2.0, mole_fractions_design)
        q_target_design = (m_dot_fg_design * (N_TRAINS_FITTED / N_GAS_TRAINS)
                           * cp_design * (t_in_design - t_target))  # design duty, all FITTED trains
        c_gas_design = (m_dot_fg_design / N_GAS_TRAINS) * cp_design  # per duct

        # --- STEP 2: loop return temperature (an INPUT, see module docstring) ---
        # Preferred path: supplied by common/loop_optimisation.py, which
        # selects it against a real objective. The fallback MINIMISES it,
        # which is not optimised - retained only so this module still runs
        # standalone, and warned about rather than silently used.
        if loop_return_c_override is not None:
            loop_return_c = (loop_return_c_override[t_target]
                             if isinstance(loop_return_c_override, dict) else float(loop_return_c_override))
        else:
            print(f"  [warning] target {t_target}: no loop_return_c supplied, falling back to the legacy "
                  f"MINIMISED derivation. Not optimised; not comparable with "
                  f"optimised runs.")
            loop_return_c = _legacy_minimised_loop_return(t_target, q_target_design)
            if loop_return_c is None:
                print(f"=== target {t_target}°c: ORC design-point solve infeasible (T_evap<=Tc) - skipping track ===\n")
                continue

        is_pinched_out = t_target <= (loop_return_c + 1.0)
        if is_pinched_out:
            print(f"=== target {t_target}°c: pinched out (loop return {loop_return_c:.1f}C) - skipping track ===\n")
            continue

        t_water_out_design_c = t_target - LOSS_BUFFER_DELTA_T
        if t_water_out_design_c <= loop_return_c:
            print(f"=== target {t_target}°c: no water-side temperature lift available - skipping track ===\n")
            continue

        # --- STEP 3: size the exchanger once at the design point ---
        # The sizing chain (loop_return -> m_dot -> UA -> area -> rows)
        # now lives in common/fgche_sizing.py and is shared with
        # common/loop_optimisation.py, so the design the optimiser scored
        # and the hardware built here cannot diverge. It also carries the
        # per-duct/TOTAL convention.
        gas_transport_design = gas_side_heat_transfer(t_in_design, m_dot_fg_design, p_gauge_design, dust_design, a_flow_minimum)

        sized = size_fgche(loop_return_c, t_water_out_design_c, q_target_design, t_in_design,
                            c_gas_design, gas_transport_design, n_tubes_per_row, DUCT_HEIGHT_M)
        if sized is None:
            print(f"=== target {t_target}°c: no water-side temperature lift available - skipping track ===\n")
            continue

        m_dot_water_fixed = sized["m_dot_loop_kg_s"]      # kg/s, both ducts
        c_water_design = sized["c_water_total_kw_per_k"]  # kW/K, both ducts
        area_316_fixed = sized["area_316_m2"]
        area_904_fixed = sized["area_904_m2"]
        rows_316_fixed = sized["rows_316"]
        rows_904_fixed = sized["rows_904"]

        # --- STEP 4: rate the fixed HX (316L reference) against every row's actual conditions ---
        row_q_actual = []
        row_t_gas_out = []
        row_t_water_out = []
        row_heat_to_downstream_kw = []
        row_bp_316, row_bp_904 = [], []
        row_fan_316, row_fan_904 = [], []
        row_pump = []
        row_corr = []
        row_eps, row_ntu = [], []

        for idx, row in df_master.iterrows():
            precomp = row_combustion[idx]
            t_in = row['FF_Outlet_Temp_Avg_C']
            m_dot_fg_total = row['Total gas flow kg/s']
            p_gauge = (row['Fabric Filter Outlet Pressure (kPa) A side'] +
                       row['Fabric Filter Outlet Pressure (kPa) B side']) / 2.0
            dust_burden = (row['A Dust Burden (mg/M3)'] + row['B Dust Burden (mg/M3)']) / 2.0

            if precomp is None or t_in <= loop_return_c:
                row_q_actual.append(0.0); row_t_gas_out.append(t_in); row_t_water_out.append(loop_return_c)
                row_heat_to_downstream_kw.append(0.0)
                row_bp_316.append(0.0); row_bp_904.append(0.0)
                row_fan_316.append(0.0); row_fan_904.append(0.0)
                row_pump.append(0.0); row_corr.append(0.0)
                row_eps.append(0.0); row_ntu.append(0.0)
                continue

            mole_fractions = {"y_co2": precomp["y_co2"], "y_h2o": precomp["y_h2o"],
                              "y_o2": precomp["y_o2"], "y_n2": precomp["y_n2"]}
            # bulk-mean Cp evaluated between inlet and the DESIGN target as
            # the best available estimate of this row's actual outlet -
            # the true outlet isn't known until after rating, but Cp's
            # sensitivity to this choice is second-order (see common/
            # gas_properties.py)
            cp_row = dynamic_flue_gas_cp((t_in + t_target) / 2.0, mole_fractions)
            c_gas_row = (m_dot_fg_total / N_GAS_TRAINS) * cp_row

            gas_transport_row = gas_side_heat_transfer(t_in, m_dot_fg_total, p_gauge, dust_burden, a_flow_minimum)
            ua_row_316_kw_per_k = gas_transport_row["u_316"] * (area_316_fixed / N_TRAINS_FITTED) / 1000.0

            rating = rate_fixed_hx(ua_row_316_kw_per_k, c_gas_row, c_water_design / N_TRAINS_FITTED, t_in, loop_return_c)

            q_actual_duct = rating["q_actual_kw"]
            q_actual_total = q_actual_duct * N_TRAINS_FITTED
            heat_to_downstream_kw = q_actual_total * HX_SHELL_EFFICIENCY

            t_adp = precomp["adp_c"]
            r_corrosion = calculate_corrosion_index(t_adp, rating["t_gas_out_c"])

            # fan power is evaluated against THIS row's own gas transport
            # (flow, density, velocity all vary with load), not the design
            # row's - the row count is what the design fixes
            fan_316 = gas_side_fan_power(rows_316_fixed, gas_transport_row)
            fan_904 = gas_side_fan_power(rows_904_fixed, gas_transport_row)
            dp_pa_316, fan_kw_316 = fan_316["dp_pa"], fan_316["fan_kw"]
            dp_pa_904, fan_kw_904 = fan_904["dp_pa"], fan_904["fan_kw"]

            pump_power_penalty_kw = loop_pump_power_kw(m_dot_water_fixed)

            row_q_actual.append(q_actual_total)
            row_t_gas_out.append(rating["t_gas_out_c"])
            row_t_water_out.append(rating["t_water_out_c"])
            row_heat_to_downstream_kw.append(heat_to_downstream_kw)
            row_bp_316.append(dp_pa_316 / 1000.0)
            row_bp_904.append(dp_pa_904 / 1000.0)
            row_fan_316.append(fan_kw_316)
            row_fan_904.append(fan_kw_904)
            row_pump.append(pump_power_penalty_kw)
            row_corr.append(r_corrosion)
            row_eps.append(rating["effectiveness"])
            row_ntu.append(rating["ntu"])

        df_output = df_master.drop(columns=['timestamp_clean']).copy()
        df_output['design_target_c'] = t_target
        df_output['loop_return_c'] = loop_return_c
        df_output['loop_mass_flow_kg_s'] = m_dot_water_fixed   # fixed design flow, all trains
        df_output['fixed_area_316_m2'] = area_316_fixed
        df_output['fixed_area_904_m2'] = area_904_fixed
        df_output['recovered_energy_kw'] = row_q_actual
        df_output['actual_gas_outlet_c'] = row_t_gas_out
        df_output['actual_water_outlet_c'] = row_t_water_out
        df_output['heat_delivered_to_downstream_kw'] = row_heat_to_downstream_kw
        df_output['backpressure_316_kpa'] = row_bp_316
        df_output['backpressure_904_kpa'] = row_bp_904
        df_output['fan_parasitic_316_kw'] = row_fan_316
        df_output['fan_parasitic_904_kw'] = row_fan_904
        df_output['water_pump_parasitic_kw'] = row_pump
        df_output['combined_corrosion_index'] = row_corr
        df_output['hx_effectiveness'] = row_eps
        df_output['hx_ntu'] = row_ntu

        arch_label = f"{active_architecture}_{ro_membrane_case}" if active_architecture == "orc_ro" else active_architecture
        output_filename = str(Path(output_dir) / f"gps_module3_output_{arch_label}_{t_target}.csv")
        df_output.to_csv(output_filename, index=False)

        df_op = df_output[df_output['recovered_energy_kw'] > 0]
        print(f"=== target {t_target}°c ({label} track, design row idx={design_idx}, loop_return={loop_return_c:.1f}C) ===")
        print(f"  fixed area (316L/904L): {area_316_fixed:.1f} / {area_904_fixed:.1f} m2 ({rows_316_fixed:.0f}/{rows_904_fixed:.0f} rows)")
        if len(df_op) > 0:
            print(f"  mean actual gas outlet temp: {df_op['actual_gas_outlet_c'].mean():.2f}C (design target was {t_target}C)")
            print(f"  mean HX effectiveness: {df_op['hx_effectiveness'].mean():.3f}")
            print(f"  mean heat delivered downstream: {df_op['heat_delivered_to_downstream_kw'].mean():.2f} kw")
            print(f"  mean auxiliary water pump load: {df_op['water_pump_parasitic_kw'].mean():.2f} kw\n")
        else:
            print("  no active rows at this track\n")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    execute_master_module3_pipeline(str(data_dir / "GPS_Operating_Data_Clean.csv"),
                                     str(data_dir / "GPS_Water_related_data.csv"),
                                     str(output_dir))
