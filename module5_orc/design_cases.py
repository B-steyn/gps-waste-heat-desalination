"""
The design cases shared by the optimiser, Stage 4 and the checks: the duct
cases, the ORC architectures with their condensing temperatures, and the
design record.

The DESIGN RECORD is the operating record with the largest flue-gas flow.
Every exchanger is sized and priced there (the full-load point); the fan
cap is checked against every record, and annual energy is the mean over the
records (module6_lcow.heat_side_annual_kw).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from common.coal_combustion import calculate_dynamic_adp, combustion_mole_balance
from common.constants import (
    P_AMBIENT_KPA, ORC_RO_TC_CURRENT_MEMBRANE_C, ORC_RO_TC_ADVANCED_MEMBRANE_C,
    ORC_MED_TC_CANDIDATES_C,
)

# (label, width_m, height_m). The measured duct, 5,790 x 4,235 mm = 24.52 m2
# at the fabric-filter outlet, is the reported case. The others bracket it as
# a sensitivity, spanning plausible flue-duct face velocities (about 4-13 m/s
# at the design record); 5.5 x 4.0 m was the working estimate before the
# duct was measured.
DUCT_CASES = [
    ("5.0x3.5", 5.0, 3.5),
    ("5.5x4.0 (superseded estimate)", 5.5, 4.0),
    ("5.79x4.235 (MEASURED)", 5.79, 4.235),
    ("6.5x5.0", 6.5, 5.0),
    ("8.0x6.0", 8.0, 6.0),
]

ARCHITECTURES = {
    "orc_ro_current": ORC_RO_TC_CURRENT_MEMBRANE_C,
    "orc_ro_advanced": ORC_RO_TC_ADVANCED_MEMBRANE_C,
}
# ORC+MED at every candidate condensing temperature. The first keeps the key
# "orc_med"; the rest carry their Tc in the key.
for _tc in ORC_MED_TC_CANDIDATES_C:
    ARCHITECTURES["orc_med" if _tc == ORC_MED_TC_CANDIDATES_C[0] else f"orc_med_tc{_tc:.0f}"] = _tc


def load_design_row(data_dir):
    """The design record: flow state and gas composition at the largest flue-gas flow."""
    coal_moles = calculate_dynamic_adp()
    df_gas = pd.read_csv(Path(data_dir) / "GPS_Operating_Data_Clean.csv")
    df_sea = pd.read_csv(Path(data_dir) / "GPS_Water_related_data.csv")
    for d in (df_gas, df_sea):
        d.columns = d.columns.str.strip()
    df_gas["k"] = df_gas["Timestamp"].astype(str).str.strip()
    df_sea["k"] = df_sea["Timestamp"].astype(str).str.strip()
    dfm = pd.merge(df_gas, df_sea[["k", "Sea water intake temp"]], on="k", how="inner")

    combustion = []
    for _, row in dfm.iterrows():
        o2 = (row["A side flue gas o2 %"] + row["B side flue gas o2 %"]) / 2.0
        pg = (row["Fabric Filter Outlet Pressure (kPa) A side"] +
              row["Fabric Filter Outlet Pressure (kPa) B side"]) / 2.0
        combustion.append(None if (row["Total gas flow kg/s"] <= 1.0 or o2 >= 21.0)
                          else combustion_mole_balance(o2, coal_moles, pg, P_AMBIENT_KPA))

    idx = dfm.loc[[c is not None for c in combustion], "Total gas flow kg/s"].idxmax()
    row = dfm.loc[idx]
    pre = combustion[idx]
    return {
        "idx": idx,
        "t_in_c": row["FF_Outlet_Temp_Avg_C"],
        "m_dot_fg": row["Total gas flow kg/s"],
        "p_gauge": (row["Fabric Filter Outlet Pressure (kPa) A side"] +
                    row["Fabric Filter Outlet Pressure (kPa) B side"]) / 2.0,
        "dust": (row["A Dust Burden (mg/M3)"] + row["B Dust Burden (mg/M3)"]) / 2.0,
        "mole_fractions": {"y_co2": pre["y_co2"], "y_h2o": pre["y_h2o"],
                            "y_o2": pre["y_o2"], "y_n2": pre["y_n2"]},
    }
