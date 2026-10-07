"""
ID fan power margin: converts the existing fan's spare capacity into a
hard cap on exchanger depth.

This is the model's only HARD FEASIBILITY constraint - everything else is
a cost trade the LCOW objective can smooth over. If the added gas-side
pressure drop pushes the fan past its motor rating, the answer is not a
more expensive design, it is a different piece of plant.

Hardware: Siemens 1RQ 1802-8Js70-Z, 6.6 kV, 2,450 kW, single-speed
induction. ONE PER DUCT, two ducts per unit.

WHY A ROW CAP RATHER THAN A PRESSURE CAP. Added fan power is LINEAR in
row count:

    dP     = rows x K x foul x (0.5 rho v_max^2)
    P_add  = V_duct x dP / eta_fan / n_fans_per_duct

so the largest feasible row count has a closed form per operating row,
and the binding constraint is simply the worst of the 461. No search is
needed, and the result is exact rather than a bisection tolerance.

WHAT THIS DOES NOT DO. It compares shaft power against motor rating. It
does NOT check that the fan WHEEL can generate the required head at the
required flow - that needs a manufacturer's curve, the system resistance
characteristic and the existing operating point, none of which exist for
this study. That boundary is deliberate: a single-speed fan absorbs added
resistance by opening its inlet vanes, and the logged 35-63% vane position
says there is head in reserve, but "in reserve" is not a number this study
can put a bound on.
"""

import numpy as np

from common.constants import (
    N_FANS_PER_DUCT, TUBE_ROW_DP_FACTOR, FAN_EFFICIENCY,
    ID_FAN_RATING_KW, ID_FAN_VOLTAGE_V, ID_FAN_MOTOR_EFFICIENCY,
)

# Design basis. There is no logged motor kW in the station data, only
# ID Fan A/B current, so shaft power is inferred from
# current. Working back from amps at PF 0.83-0.88 implies a fan static
# rise of 5.7-6.0 kPa against a measured -4.45 kPa suction, which is
# right for an ID fan of this duty. That confirms the RANGE is physical;
# it does not single out 0.85. Treat 0.85 as a defensible central
# assumption and the band as a reported sensitivity.
ID_FAN_POWER_FACTOR = 0.85
ID_FAN_PF_BAND = (0.83, 0.88)

# Gas-side fouling applied as a multiplier on tube-bank dP. Clean (1.0) is
# the reported design; 1.2 and 1.4 are the fouled cases, which cost
# roughly 5 and 8 C of extraction track at PF 0.88.
FOULING_DP_FACTORS = (1.0, 1.2, 1.4)


def existing_fan_shaft_kw(amps, power_factor=ID_FAN_POWER_FACTOR):
    """Shaft power of one ID fan from its logged line current."""
    return (np.sqrt(3.0) * ID_FAN_VOLTAGE_V * np.asarray(amps, dtype=float)
            * power_factor * ID_FAN_MOTOR_EFFICIENCY) / 1000.0


def added_fan_kw_per_row(rows, rho_gas, v_max, volumetric_flow_m3s_duct,
                          fouling_factor=1.0):
    """Added shaft power PER FAN for a bank `rows` deep, at one row's flow."""
    dp_pa = rows * TUBE_ROW_DP_FACTOR * fouling_factor * (0.5 * np.asarray(rho_gas) * np.asarray(v_max) ** 2)
    return (np.asarray(volumetric_flow_m3s_duct) * dp_pa / FAN_EFFICIENCY
            / 1000.0 / N_FANS_PER_DUCT)


def max_feasible_rows(amps, rho_gas, v_max, volumetric_flow_m3s_duct,
                       power_factor=ID_FAN_POWER_FACTOR, fouling_factor=1.0,
                       rating_kw=ID_FAN_RATING_KW, utilisation=1.0):
    """
    Largest integer bank depth for which NO operating row exceeds the
    motor rating.

    `utilisation` < 1.0 reserves margin (e.g. 0.95 to hold 5% back).
    Returns (max_rows, diagnostics). max_rows of 0 means the fan has no
    usable headroom at all on at least one row.
    """
    existing = existing_fan_shaft_kw(amps, power_factor)
    budget = rating_kw * utilisation - existing            # kW available per row
    unit_add = added_fan_kw_per_row(1.0, rho_gas, v_max, volumetric_flow_m3s_duct,
                                     fouling_factor)       # kW per row of depth

    with np.errstate(divide="ignore", invalid="ignore"):
        per_row_limit = np.where(unit_add > 0, budget / unit_add, np.inf)

    binding = int(np.nanargmin(per_row_limit))
    max_rows = int(np.floor(np.nanmin(per_row_limit)))
    return max(max_rows, 0), {
        "binding_row_index": binding,
        "existing_kw_at_binding": float(existing[binding]),
        "existing_kw_max": float(np.nanmax(existing)),
        "headroom_kw_at_binding": float(budget[binding]),
        "power_factor": power_factor,
        "fouling_factor": fouling_factor,
    }


def load_fan_rows(data_dir):
    """Per-row flow state and logged fan current, for the margin check."""
    import pandas as pd
    from pathlib import Path
    df = pd.read_csv(Path(data_dir) / "GPS_Operating_Data_Clean.csv")
    df.columns = df.columns.str.strip()
    df = df[df["Total gas flow kg/s"] > 1.0]
    return {
        "t_in_c": df["FF_Outlet_Temp_Avg_C"].values,
        "m_dot_fg": df["Total gas flow kg/s"].values,
        "p_gauge": ((df["Fabric Filter Outlet Pressure (kPa) A side"]
                     + df["Fabric Filter Outlet Pressure (kPa) B side"]) / 2.0).values,
        "dust": ((df["A Dust Burden (mg/M3)"] + df["B Dust Burden (mg/M3)"]) / 2.0).values,
        "amps": pd.to_numeric(df["ID Fan A amps"], errors="coerce").values,
    }


def operating_record_factors(data_dir):
    """
    Annual-mean heat-side loads as fractions of their design-record values.

    Every design is sized and priced at the DESIGN RECORD, the record with the
    largest gas flow, and the fan cap checks the worst record. Annual energy is
    a different question. The station spends most of the year below that flow,
    so the electricity the heat side uses or makes is the mean over the
    operating records, each weighted equally, as the RO preheat already is.

      fan   mean added fan power / design-record added fan power. For a fixed
            bank, gas_side_fan_power is proportional to rows x m^3 / rho^2, so
            this fraction is the same for every design, duct and row count.
      heat  mean gas flow / design-record gas flow. The heat reaching the
            downstream plant is scaled linearly with gas flow, so an ORC at
            constant cycle efficiency makes this fraction of its design
            output.

    ASSUMPTION, stated: the fan efficiency is the same at every load. These
    fans are single-speed with inlet-vane control, which loses efficiency at
    part load, so the fan fraction is a central estimate, not a bound
    (validation/independent_checks.py estimates the effect from the fans'
    own logged current).
    """
    from common.constants import PITCH_RATIO, DUCT_WIDTH_M
    from common.hx_loop_physics import duct_geometry, gas_side_heat_transfer
    from common.fgche_sizing import gas_side_fan_power

    rows = load_fan_rows(data_dir)
    _, a_flow = duct_geometry(DUCT_WIDTH_M, PITCH_RATIO)
    gt = gas_side_heat_transfer(rows["t_in_c"], rows["m_dot_fg"], rows["p_gauge"],
                                 rows["dust"], a_flow)
    fan = gas_side_fan_power(1.0, gt)["fan_kw"]
    i = int(np.argmax(rows["m_dot_fg"]))          # the design record: maximum gas flow
    return {"fan": float(np.mean(fan) / fan[i]),
            "heat": float(np.mean(rows["m_dot_fg"]) / rows["m_dot_fg"][i]),
            "n_records": int(len(fan)),
            "design_m_dot_fg": float(rows["m_dot_fg"][i])}


def row_caps_for_ducts(data_dir, duct_cases, power_factor=ID_FAN_POWER_FACTOR,
                        fouling_factor=1.0, utilisation=1.0):
    """
    Maximum feasible bank depth for each swept duct case.

    Computed once here and shared by every driver, so that no two copies
    of the same physics can drift apart.
    """
    from common.constants import PITCH_RATIO, DUCT_HEIGHT_M
    from common.hx_loop_physics import duct_geometry, gas_side_heat_transfer

    rows = load_fan_rows(data_dir)
    caps = {}
    for label, dw, dh in duct_cases:
        n_tubes, a_ref = duct_geometry(dw, PITCH_RATIO)
        a_flow = a_ref * (dh / DUCT_HEIGHT_M)
        gts = [gas_side_heat_transfer(t, m, p, du, a_flow) for t, m, p, du in
               zip(rows["t_in_c"], rows["m_dot_fg"], rows["p_gauge"], rows["dust"])]
        cap, diag = max_feasible_rows(
            rows["amps"], [g["rho_gas"] for g in gts], [g["v_max"] for g in gts],
            [g["volumetric_flow_m3s_duct"] for g in gts],
            power_factor=power_factor, fouling_factor=fouling_factor,
            utilisation=utilisation)
        caps[label] = cap
    return caps
