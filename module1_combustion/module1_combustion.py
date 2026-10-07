"""
Module 1: per-record combustion. For each operating record it solves the
coal's combustion mole balance at the measured O2, giving the excess air,
flue-gas composition and acid dew point, then the composition-weighted gas
heat capacity and the heat recovered by cooling the gas to each extraction
track.

The exchanger sits downstream of the fabric filter, where the measured gas
is about 93-125 C against a site acid dew point of about 105 C, so it runs
below the dew point at typical and low loads by design: see TAP_POINT_NOTE
in common/constants.py and calculate_corrosion_index() in
common/coal_combustion.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.coal_combustion import calculate_dynamic_adp, combustion_mole_balance, calculate_corrosion_index
from common.gas_properties import dynamic_flue_gas_cp
from common.constants import P_AMBIENT_KPA, T_AMBIENT_CELSIUS, TAP_POINT_NOTE, TARGET_TEMPERATURES

# every extraction track the model evaluates
TRACKS = tuple(sorted(TARGET_TEMPERATURES, reverse=True))


def run_module1(input_csv, output_csv):
    """
    Adds the combustion results, acid dew point, heat capacities and the
    heat recovered at every track in TRACKS to each operating record, and
    writes the result to output_csv.
    """
    coal_moles = calculate_dynamic_adp()

    df = pd.read_csv(input_csv)
    # Header whitespace differs between exports ("... A side" vs "... A side ");
    # strip once so the lookups below cannot silently depend on it.
    df.columns = df.columns.str.strip()

    cols = {k: [] for k in ("dynamic_lambda", "flue_gas_y_h2o", "flue_gas_y_so3_ppm",
                            "calculated_adp_c", "dynamic_cp_dead_state")}
    for t in TRACKS:
        for k in (f"dynamic_cp_{t:.0f}", f"heat_recovered_{t:.0f}_kw", f"corrosion_index_{t:.0f}"):
            cols[k] = []

    for idx, row in df.iterrows():
        o2_meas = (row['A side flue gas o2 %'] + row['B side flue gas o2 %']) / 2.0
        p_gauge = (row['Fabric Filter Outlet Pressure (kPa) A side'] +
                   row['Fabric Filter Outlet Pressure (kPa) B side']) / 2.0
        t_in = row['FF_Outlet_Temp_Avg_C']
        m_dot_fg = row['Total gas flow kg/s']

        if o2_meas >= 21.0 or m_dot_fg <= 0:
            for k in cols:
                cols[k].append(45.0 if k == "calculated_adp_c" else 0.0)
            continue

        result = combustion_mole_balance(o2_meas, coal_moles, p_gauge, P_AMBIENT_KPA)
        mole_fractions = {"y_co2": result["y_co2"], "y_h2o": result["y_h2o"],
                          "y_o2": result["y_o2"], "y_n2": result["y_n2"]}

        cols["dynamic_lambda"].append(result["lambda"])
        cols["flue_gas_y_h2o"].append(result["y_h2o"])
        cols["flue_gas_y_so3_ppm"].append(result["so3_ppm"])
        cols["calculated_adp_c"].append(result["adp_c"])
        # Cp at the bulk mean of each span (dead state for Module 2's
        # baselines, each track for the recovered band) - common/gas_properties.py
        cols["dynamic_cp_dead_state"].append(
            dynamic_flue_gas_cp((t_in + T_AMBIENT_CELSIUS) / 2.0, mole_fractions))
        for t in TRACKS:
            cp_t = dynamic_flue_gas_cp((t_in + t) / 2.0, mole_fractions)
            cols[f"dynamic_cp_{t:.0f}"].append(cp_t)
            cols[f"heat_recovered_{t:.0f}_kw"].append(m_dot_fg * cp_t * (t_in - t) if t_in > t else 0.0)
            cols[f"corrosion_index_{t:.0f}"].append(calculate_corrosion_index(result["adp_c"], t))

    for k, v in cols.items():
        df[k] = v

    print("Operating diagnostics:")
    print(f"mean excess air multiplier: {df['dynamic_lambda'].mean():.3f}")
    print(f"mean flue gas water vapor: {df['flue_gas_y_h2o'].mean() * 100:.2f}%")
    print(f"mean residual gaseous so3: {df['flue_gas_y_so3_ppm'].mean():.4f} ppm")
    print(f"mean chemical acid dew point: {df['calculated_adp_c'].mean():.2f} C")
    for t in TRACKS:
        print(f"--- {t:.0f} C track ---")
        print(f"  mean recovered heat: {df[f'heat_recovered_{t:.0f}_kw'].mean():.2f} kw")
        print(f"  peak recovered heat: {df[f'heat_recovered_{t:.0f}_kw'].max():.2f} kw")

    df.to_csv(output_csv, index=False)
    print(f"module 1 output exported to: {output_csv}\n")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    run_module1(str(data_dir / "GPS_Operating_Data_Clean.csv"), str(output_dir / "GPS_Module1_Output.csv"))
