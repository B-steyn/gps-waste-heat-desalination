"""
Module 2b: component-level exergy destruction (Gouy-Stodola) for the
flue-gas exchanger (FGCHE) and intermediate loop, per operating record,
reported in kW and as a fraction of the exergy entering. Module 2 gives the
idealised exergy content of the recovered heat band; this pass needs
two-sided state points (flue gas AND loop water) that exist only once
Module 3 has rated the exchanger, so it runs after Module 3.

Scope: the FGCHE and loop, with the ID fan and loop pump work, at the
Direct-MED exchanger Module 3 rates per track. The design-record balances
of all four reported designs, including the ORC and the RO preheater, are
in analysis/energy_tables.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.coal_combustion import calculate_dynamic_adp, combustion_mole_balance
from common.gas_properties import dynamic_flue_gas_cp
from common.exergy_analysis import (hx_exergy_destruction, parasitic_exergy_destruction,
                                    gas_exergy_and_entropy_change, water_exergy_gain)
from common.fluid_properties import ThermoCore
from common.constants import (
    P_AMBIENT_KPA, T_AMBIENT_CELSIUS, P_WATER_LOOP_PA, LOSS_BUFFER_DELTA_T,
    HX_SHELL_EFFICIENCY, ARCHITECTURE_WATER_RETURN_C, TARGET_TEMPERATURES,
)


def run_exergy_destruction(module3_output_dir, architecture="direct_med"):
    coal_moles = calculate_dynamic_adp()
    thermo = ThermoCore(T0_celsius=T_AMBIENT_CELSIUS, P0_kpa=P_AMBIENT_KPA)

    for t_target in TARGET_TEMPERATURES:
        input_csv = Path(module3_output_dir) / f"gps_module3_output_{architecture}_{t_target}.csv"
        print(f"loading module 3 output for {t_target}°c track: {input_csv.name}...")
        df = pd.read_csv(input_csv)

        if len(df) == 0:
            print(f"[warning]: {input_csv.name} has no rows - skipping track.\n")
            continue

        # Per-record states come from Module 3's RATING, not the design point:
        # Module 3 rates a FIXED exchanger at a FIXED loop flow, so both outlet
        # temperatures move record to record (the 85 C design runs at about a
        # 79.5 C mean gas outlet), and the loop return is the design's.
        for col in ("loop_return_c", "actual_water_outlet_c", "actual_gas_outlet_c",
                    "loop_mass_flow_kg_s"):
            if col not in df.columns:
                raise KeyError(f"{input_csv.name} lacks '{col}' - regenerate Module 3 first")

        ex_destroyed_hx = []
        ex_destroyed_fan = []
        ex_destroyed_pump = []
        ex_destroyed_total = []
        ex_entering_total = []
        ex_destruction_fraction = []
        exergy_efficiency_hx = []
        ex_destroyed_loss = []
        balance_resid = []

        for idx, row in df.iterrows():
            t_in = row['FF_Outlet_Temp_Avg_C']
            m_dot_fg_total = row['Total gas flow kg/s']
            o2_meas = (row['A side flue gas o2 %'] + row['B side flue gas o2 %']) / 2.0
            p_gauge = (row['Fabric Filter Outlet Pressure (kPa) A side'] +
                       row['Fabric Filter Outlet Pressure (kPa) B side']) / 2.0
            q_recovered_total = row['recovered_energy_kw']
            q_absorbed_water = row['heat_delivered_to_downstream_kw']
            fan_kw = row['fan_parasitic_316_kw']  # 316L as the representative track
            pump_kw = row['water_pump_parasitic_kw']

            if q_recovered_total <= 0:
                ex_destroyed_hx.append(0.0)
                ex_destroyed_fan.append(0.0)
                ex_destroyed_pump.append(0.0)
                ex_destroyed_total.append(0.0)
                ex_entering_total.append(0.0)
                ex_destruction_fraction.append(0.0)
                exergy_efficiency_hx.append(0.0)
                ex_destroyed_loss.append(0.0)
                balance_resid.append(0.0)
                continue

            combustion = combustion_mole_balance(o2_meas, coal_moles, p_gauge, P_AMBIENT_KPA)
            mole_fractions = {"y_co2": combustion["y_co2"], "y_h2o": combustion["y_h2o"],
                              "y_o2": combustion["y_o2"], "y_n2": combustion["y_n2"]}
            # Cp on the SAME basis Module 3 rated with, (t_in + design target)/2,
            # so the gas heat released matches Module 3's rated duty exactly
            cp_flue_gas = dynamic_flue_gas_cp((t_in + t_target) / 2.0, mole_fractions)
            t_gas_out = row['actual_gas_outlet_c']
            t_water_in = row['loop_return_c']
            t_water_out = row['actual_water_outlet_c']
            m_dot_water = row['loop_mass_flow_kg_s']
            # The rated water states carry the FULL gas duty: the model applies
            # HX_SHELL_EFFICIENCY DOWNSTREAM of the exchanger, between it and its
            # consumer (fgche_sizing, Module 3). So the exchanger control volume
            # takes NO shell-loss term (one would break its energy balance by the
            # 1.5 % loss). The loop heat loss is instead destroyed separately, at
            # the loop's mean temperature: E_D = Q_loss (1 - T0 / T_loop).
            hx = hx_exergy_destruction(m_dot_fg_total, cp_flue_gas, t_in, t_gas_out,
                                        m_dot_water, t_water_in, t_water_out,
                                        0.0, thermo, P_WATER_LOOP_PA)
            q_loss = q_recovered_total * (1.0 - HX_SHELL_EFFICIENCY)
            t_loop_k = 0.5 * (t_water_in + t_water_out) + 273.15
            e_d_loss = q_loss * (1.0 - (T_AMBIENT_CELSIUS + 273.15) / t_loop_k)

            # first-law check on the CV: gas heat released vs water heat gained
            q_gas = m_dot_fg_total * cp_flue_gas * (t_in - t_gas_out)
            q_water = water_exergy_gain(thermo, m_dot_water, t_water_in, t_water_out, P_WATER_LOOP_PA)[2]
            resid = 100.0 * (q_gas - q_water) / q_gas if q_gas > 0 else 0.0

            e_d_fan = parasitic_exergy_destruction(fan_kw)
            e_d_pump = parasitic_exergy_destruction(pump_kw)
            e_d_total = hx["exergy_destroyed_kw"] + e_d_loss + e_d_fan + e_d_pump

            # total exergy entering the FGCHE at T_in, relative to dead
            # state - the denominator for the exergy destruction fraction
            cp_dead_state = dynamic_flue_gas_cp((t_in + T_AMBIENT_CELSIUS) / 2.0, mole_fractions)
            ex_entering, _, _ = gas_exergy_and_entropy_change(m_dot_fg_total, cp_dead_state, t_in, T_AMBIENT_CELSIUS)

            ex_destroyed_hx.append(hx["exergy_destroyed_kw"])
            ex_destroyed_loss.append(e_d_loss)
            balance_resid.append(resid)
            ex_destroyed_fan.append(e_d_fan)
            ex_destroyed_pump.append(e_d_pump)
            ex_destroyed_total.append(e_d_total)
            ex_entering_total.append(ex_entering)
            ex_destruction_fraction.append((e_d_total / ex_entering * 100.0) if ex_entering > 0 else 0.0)
            exergy_efficiency_hx.append(
                (hx["exergy_gain_water_kw"] / hx["exergy_given_up_by_gas_kw"] * 100.0)
                if hx["exergy_given_up_by_gas_kw"] > 0 else 0.0)

        df_output = df.copy()
        df_output['exergy_destroyed_hx_kw'] = ex_destroyed_hx
        df_output['exergy_destroyed_loop_loss_kw'] = ex_destroyed_loss
        df_output['hx_energy_balance_residual_pct'] = balance_resid
        df_output['exergy_destroyed_fan_kw'] = ex_destroyed_fan
        df_output['exergy_destroyed_pump_kw'] = ex_destroyed_pump
        df_output['exergy_destroyed_total_kw'] = ex_destroyed_total
        df_output['exergy_entering_kw'] = ex_entering_total
        df_output['exergy_destruction_fraction_pct'] = ex_destruction_fraction
        df_output['hx_exergy_efficiency_pct'] = exergy_efficiency_hx

        output_filename = str(Path(module3_output_dir) / f"gps_module2b_exergy_destruction_{architecture}_{t_target}.csv")
        df_output.to_csv(output_filename, index=False)

        df_active = df_output[df_output['exergy_entering_kw'] > 0]
        print(f"--- exergy destruction summary for track target {t_target}°c ---")
        if len(df_active) > 0:
            print(f"  mean exergy destroyed (HX): {df_active['exergy_destroyed_hx_kw'].mean():.2f} kw")
            print(f"  mean exergy destroyed (fan+pump): {(df_active['exergy_destroyed_fan_kw'] + df_active['exergy_destroyed_pump_kw']).mean():.2f} kw")
            print(f"  mean total exergy destroyed: {df_active['exergy_destroyed_total_kw'].mean():.2f} kw")
            print(f"  exchanger energy balance residual: max |{df_active['hx_energy_balance_residual_pct'].abs().max():.3f}|%")
            print(f"  mean exergy destruction fraction (of exergy entering): {df_active['exergy_destruction_fraction_pct'].mean():.2f}%")
            print(f"  mean HX exergy efficiency (water gain / gas given up): {df_active['hx_exergy_efficiency_pct'].mean():.2f}%\n")
        else:
            print("  no active rows at this track\n")


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    run_exergy_destruction(str(output_dir))
