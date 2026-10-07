"""
Module 6: levelised cost of water (LCOW) for every architecture and track.

Pulls together, per architecture and per track:
  Module 3  -> flue-gas exchanger (FGCHE) area, ID fan and loop pump loads, heat delivered
  Module 5  -> ORC net output (path A or B) and condenser heat
  Module 4a / common.med -> MED water production
  common.ro_thermal      -> RO water production and specific energy (SEC)
then applies common/costing.py and common/lcow.py.

Conventions:

  REAL discount basis. Costs are constant 2026 AUD with no forward
  escalation, so the rate must be real; the nominal 8 % would overstate the
  capital charge by about 27 %.

  ONE cost basis for MED and RO: Bhojwani et al. (2019), with MED at 1.25x
  SWRO at matched capacity, by construction.

  ARCHITECTURE-DEPENDENT operating hours. MED is heat-driven and capped by
  the station's availability (76.2 %); RO runs through boiler outages on
  grid power at its own 95 % availability, at a higher SEC in those hours.

  P90 MED sizing. MED is sized for the 90th-percentile hour of heat, and its
  annual water carries the resulting capacity factor.

  FULL electricity balance: ORC output less the added ID fan, loop pump AND
  desalination process demand. The heat-side terms count only in the hours
  the station makes flue gas, at their MEAN over the operating records
  (heat_side_annual_kw); the process demand counts in all of RO's hours.

Architectures: Direct-MED, ORC+MED, ORC+RO (current 38 C and advanced 50 C
membranes), Direct-RO (heat recovered, no ORC) and conventional SWRO (no
heat recovery, grid-powered, capacity held at the largest RO plant's).
Direct-RO and conventional SWRO are comparison cases: together they split
the cost of ORC+RO into the cost of recovering the heat and the cost of the
ORC.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from common.costing import (
    hx_purchased_cost_aud, installed_cost, orc_cost_aud, med_cost_aud, swro_cost_aud,
    fgche_cost_aud, orc_exchanger_cost_aud, med_capex_gor_factor,
)
from common.lcow import (
    levelised_cost_of_water, annual_operating_hours, blended_sec_kwh_m3,
    capital_recovery_factor,
)
from common.ro_thermal import ro_specific_energy
from common.constants import (
    N_TRAINS_FITTED, N_GAS_TRAINS, DUCT_WIDTH_M, DUCT_HEIGHT_M, PITCH_RATIO,
    HX_SHELL_EFFICIENCY, LOSS_BUFFER_DELTA_T, ORC_CONDENSER_PINCH_C, ORC_EVAPORATOR_PINCH_C,
    SEAWATER_CP_KJ_KGK, ID_FAN_MOTOR_EFFICIENCY, RO_RECOVERY,
)
from common.economic_constants import (
    WACC_REAL, ELECTRICITY_AUD_PER_MWH, DESAL_COST_ANCHOR, MEMBRANE_LIFE_YEARS,
    ASSET_LIFE_HEAT_RECOVERY_YEARS, OANDM_RATE_ORC_HX,
)
from common.fgche_sizing import size_fgche, gas_side_fan_power, loop_pump_power_kw
from common.gas_properties import dynamic_flue_gas_cp
from common.hx_loop_physics import gas_side_heat_transfer, duct_geometry
from common.fan_margin import row_caps_for_ducts, operating_record_factors
from common.med import (gor_for_effects, max_feasible_effects, condenser_temperature_c,
                        distillate_m3_per_day, MED_TBT_FLOOR_C, MED_TBT_CEILING_C,
                        MED_DELTA_T_PER_EFFECT_C)

# The ID fan's row cap at the measured duct: the hard feasibility limit on
# every exchanger (common/fan_margin.py).
try:
    _MED_ROW_CAP = row_caps_for_ducts(
        str(Path(__file__).resolve().parent.parent / "data"),
        [("measured", DUCT_WIDTH_M, DUCT_HEIGHT_M)])["measured"]
except Exception:      # data not present
    _MED_ROW_CAP = None

MED_SIZING_PERCENTILE = 90          # MED is sized for the P90 hour of delivered heat
MED_PROCESS_SEC_KWH_M3 = 1.5        # MED pumping and vacuum, NOT thermal - UNCITED;
                                    # literature range about 1.5-2.5, so this is the low end
ORC_U_BASELINE_W_M2K = 1000.0       # midpoint of costing.ORC_U_BAND_W_M2K


def _record_factors(_cache={}):
    """Annual-mean fractions of the design-record fan power and heat (common.fan_margin)."""
    if not _cache:
        _cache.update(operating_record_factors(str(Path(__file__).resolve().parent.parent / "data")))
    return _cache


def heat_side_annual_kw(orc_gross_kw, fan_kw, loop_pump_kw):
    """
    The heat side's electricity, averaged over the hours with flue gas.

    `fan_kw` and `orc_gross_kw` arrive at the DESIGN record, the maximum gas
    flow. That record sizes and prices the plant and is what the fan cap
    checks, so those values keep that meaning everywhere else. Energy is the
    mean over the operating records instead:

      fan   added shaft power scales with m^3 / rho^2, so its mean is a fixed
            fraction of the design value (about 37 %). It is shaft power, so it
            is divided by the motor efficiency to give electricity.
      ORC   its heat follows gas flow, as the RO heat does, so at constant
            cycle efficiency it makes the mean flow fraction (about 69 %) of
            its design output. The package is still priced on the design
            rating.
      pump  the loop runs at constant flow, as Module 3 rates it, so the pump
            draws its design power in every hour.
    """
    f = _record_factors()
    fan_electric_kw = fan_kw * f["fan"] / ID_FAN_MOTOR_EFFICIENCY
    orc_mean_kw = orc_gross_kw * f["heat"]
    return {"orc_mean_kw": orc_mean_kw, "fan_electric_kw": fan_electric_kw,
            "heat_side_kw": orc_mean_kw - fan_electric_kw - loop_pump_kw}


def _hx_capex(area_m2, material="316L"):
    """
    FGCHE installed cost. `area_m2` is TOTAL across the fitted trains,
    costed as N_TRAINS_FITTED separate exchangers (one per duct), not one of
    N times the area. The cost correlation is non-linear, so this is about
    12 % dearer at 2,864 m2 than a single unit: the physical case and the
    conservative one.
    """
    return fgche_cost_aud(area_m2, N_TRAINS_FITTED, material=material)


def _orc_exchanger_capex(orc_evap_ua_w_per_k, q_cond_kw, t_cond_c,
                          t_sink_in_c, t_sink_out_c, u_w_m2k=ORC_U_BASELINE_W_M2K):
    """
    Installed cost of the ORC's own evaporator and condenser. Pricing the
    ORC only on A$/kW of output would extrapolate a vendor package price far
    outside the thermal scale it was observed on, at well under 1 % thermal
    efficiency; the exchangers do not shrink when turbine output does.

    LIMITATION: the condenser is in SEAWATER service, where 316L is prone to
    pitting and crevice attack and titanium would normally be specified. The
    correlation has no titanium factor, so this errs LOW on the condenser.
    """
    if not orc_evap_ua_w_per_k or orc_evap_ua_w_per_k <= 0:
        return None
    return orc_exchanger_cost_aud(
        q_evap_kw=None, q_cond_kw=q_cond_kw, evap_profile=None,
        t_cond_c=t_cond_c, t_sink_in_c=t_sink_in_c, t_sink_out_c=t_sink_out_c,
        u_w_m2k=u_w_m2k, precomputed_evap_area_m2=orc_evap_ua_w_per_k / u_w_m2k)


SEAWATER_DESIGN_PERCENTILE = 50     # median intake as the RO sizing point
# MED final-condenser design seawater: the WARMEST recorded operating hour.
# The effect count must fit the TBT-to-condenser range whenever the plant
# runs, and a warmer sea raises the condenser temperature and squeezes that
# range; sized at the median, the installed effects would not fit in 220 of
# the 461 hours. MED condensers are conventionally designed for the maximum
# cooling-water temperature. (RO keeps the median: there it sets feed
# capacity, not a physical fit.)
MED_CONDENSER_DESIGN_PERCENTILE = 100


def _seawater_rows(data_dir):
    """Per-record seawater intake temperature paired with flue-gas flow."""
    g = pd.read_csv(Path(data_dir) / "GPS_Operating_Data_Clean.csv")
    w = pd.read_csv(Path(data_dir) / "GPS_Water_related_data.csv")
    for d in (g, w):
        d.columns = d.columns.str.strip()
    g["k"] = g["Timestamp"].astype(str).str.strip()
    w["k"] = w["Timestamp"].astype(str).str.strip()
    m = pd.merge(g[g["Total gas flow kg/s"] > 1.0],
                 w[["k", "Sea water intake temp"]], on="k", how="inner")
    return m["Sea water intake temp"].values, m["Total gas flow kg/s"].values


def ro_annual_performance(q_cond_design_kw, ceiling_c, data_dir,
                           design_percentile=SEAWATER_DESIGN_PERCENTILE):
    """
    RO feed sizing and ANNUAL specific energy from the per-record seawater
    intake (20.4-29.7 C over the operating records).

    The feed flow is sized so that the design-record heat lifts it from the
    median intake to the membrane ceiling. On each record the feed then
    reaches min(T_sw + q_row / (m cp), ceiling): the flow is fixed and the
    temperature floats below the ceiling. The SEC is the mean over the
    records, preheated and (for outage hours) at the raw intake temperature.

    APPROXIMATION: the per-record heat is scaled linearly with flue-gas flow.
    Module 3's per-record epsilon-NTU rating shows effectiveness RISING as
    flow falls, so linear scaling understates part-load heat and is
    conservative on the preheat benefit.
    """
    t_sw, gas = _seawater_rows(data_dir)
    t_design = float(np.percentile(t_sw, design_percentile))
    if ceiling_c - t_design <= 0:
        return None

    feed_kg_s = q_cond_design_kw / (SEAWATER_CP_KJ_KGK * (ceiling_c - t_design))
    q_row = q_cond_design_kw * (gas / gas.max())
    t_feed = np.minimum(t_sw + q_row / (feed_kg_s * SEAWATER_CP_KJ_KGK), ceiling_c)

    sec_pre = float(np.mean([ro_specific_energy(t, r=RO_RECOVERY)["sec_kwh_m3"] for t in t_feed]))
    sec_amb = float(np.mean([ro_specific_energy(t, r=RO_RECOVERY)["sec_kwh_m3"] for t in t_sw]))

    return {
        "design_seawater_c": t_design,
        "feed_kg_s": feed_kg_s,
        "water_m3_per_day": feed_kg_s * RO_RECOVERY * 86400.0 / 1000.0,
        "sec_preheated_kwh_m3": sec_pre,
        "sec_ambient_kwh_m3": sec_amb,
        "feed_temp_mean_c": float(np.mean(t_feed)),
        "feed_temp_max_c": float(np.max(t_feed)),
        "n_rows": len(t_feed),
    }


def assemble_architecture(name, technology, design_heat_kw, water_m3_per_day,
                           hx_area_m2, fan_kw, loop_pump_kw, orc_gross_kw,
                           process_sec_kwh_m3, sec_ambient_kwh_m3=None,
                           orc_evap_ua_w_per_k=None, orc_t_cond_c=None,
                           orc_sink_in_c=None, orc_sink_out_c=None,
                           orc_u_w_m2k=ORC_U_BASELINE_W_M2K,
                           extra_heat_capex_aud=0.0, med_gor=None, capacity_factor=1.0,
                           material="316L", anchor=DESAL_COST_ANCHOR,
                           real_rate=WACC_REAL, electricity_price=ELECTRICITY_AUD_PER_MWH,
                           membrane_life_years=MEMBRANE_LIFE_YEARS):
    """
    One architecture's full LCOW build-up.

    `technology` selects the desalination cost correlation ('MED' or 'SWRO')
    AND the operating-hours rule - they are tied together deliberately,
    since MED's heat dependence is what caps its hours.

    `sec_ambient_kwh_m3` is only meaningful for RO: the SEC when the boiler
    is down and the feed is unpreheated. Pass None for MED, which simply does
    not run then.

    `orc_gross_kw` is the ORC's net output at the design record: after its
    own pump, generator and auxiliaries, before the ID fan and loop pump.
    """
    hours = annual_operating_hours(
        "DIRECT_MED" if technology.upper() == "MED" else "ORC_RO")

    # capacity_factor < 1 only for MED, whose output is limited by the heat
    # arriving each hour (see _med_capacity_factor). RO runs at capacity.
    water_per_year = water_m3_per_day * capacity_factor * hours["total_h"] / 24.0

    # blended SEC across preheated and ambient hours (RO only)
    if sec_ambient_kwh_m3 is not None and hours["ambient_h"] > 0:
        sec_effective = blended_sec_kwh_m3(process_sec_kwh_m3, sec_ambient_kwh_m3, hours)
    else:
        sec_effective = process_sec_kwh_m3

    # process electrical demand, averaged over operating hours
    process_kw = sec_effective * water_m3_per_day * capacity_factor / 24.0

    # The heat side (ORC output, added ID fan duty, loop pump) runs only while
    # the station makes flue gas: the preheated hours. RO runs through outages
    # on grid power, so for RO those are fewer than its operating hours, and
    # net_electricity_kw is an AVERAGE over operating hours, like process_kw.
    # For MED the two sets of hours are the same. Within those hours the heat
    # side is its mean over the operating records.
    heat_share = hours["preheated_h"] / hours["total_h"] if hours["total_h"] > 0 else 0.0
    hs = heat_side_annual_kw(orc_gross_kw, fan_kw, loop_pump_kw)
    heat_side_kw = hs["heat_side_kw"]
    net_electricity_kw = heat_side_kw * heat_share - process_kw

    # ORC exchangers, priced on the same correlation as the FGCHE and pooled
    # with it: they are heat-recovery equipment, stranded on station closure
    # exactly as the FGCHE is, so they carry that pool's O&M rate and asset
    # life. Reported separately below so their size stays visible.
    orc_hx = None
    if orc_evap_ua_w_per_k and orc_t_cond_c is not None:
        orc_hx = _orc_exchanger_capex(orc_evap_ua_w_per_k, design_heat_kw,
                                       orc_t_cond_c, orc_sink_in_c, orc_sink_out_c,
                                       u_w_m2k=orc_u_w_m2k)
    orc_hx_aud = orc_hx["installed_aud"] if orc_hx else 0.0

    capex = {
        "heat_recovery": (_hx_capex(hx_area_m2, material)
                          + orc_cost_aud(orc_gross_kw) + orc_hx_aud
                          + extra_heat_capex_aud),
        # MED capital carries Bhojwani's performance-ratio correction for the
        # GOR actually run (common.costing.med_capex_gor_factor), the same
        # correction the effect-count sweep applies.
        "water": ((med_cost_aud(water_m3_per_day, anchor)
                   * (med_capex_gor_factor(med_gor) if med_gor else 1.0))
                  if technology.upper() == "MED"
                  else swro_cost_aud(water_m3_per_day, anchor)),
    }

    # membranes are a replaceable subset of the RO plant, taken as the
    # commonly quoted 20 % of RO capital; MED has no equivalent
    membrane_capex = capex["water"] * 0.20 if technology.upper() != "MED" else 0.0

    result = levelised_cost_of_water(
        capex_by_pool_aud=capex, water_m3_per_year=water_per_year,
        net_electricity_kw=net_electricity_kw, operating_hours=hours["total_h"],
        membrane_capex_aud=membrane_capex, real_discount_rate=real_rate,
        electricity_aud_per_mwh=electricity_price,
        membrane_life_years=membrane_life_years)

    result.update({
        "architecture": name, "technology": technology,
        "design_heat_kw": design_heat_kw,
        "water_m3_per_day": water_m3_per_day,
        "operating_hours": hours["total_h"],
        "preheated_hours": hours["preheated_h"],
        "ambient_hours": hours["ambient_h"],
        "sec_effective_kwh_m3": sec_effective,
        "process_kw": process_kw,
        # design record (sizing, pricing, the fan cap) ...
        "orc_gross_kw": orc_gross_kw, "fan_kw": fan_kw, "loop_pump_kw": loop_pump_kw,
        # ... and the annual mean over the operating records (energy)
        "orc_mean_kw": hs["orc_mean_kw"], "fan_electric_kw": hs["fan_electric_kw"],
        "heat_side_kw": heat_side_kw,
        "net_electricity_kw": net_electricity_kw,
        "hx_area_m2": hx_area_m2,
        "capex_heat_recovery_aud": capex["heat_recovery"],
        "capex_orc_package_aud": orc_cost_aud(orc_gross_kw),
        "capex_orc_exchangers_aud": orc_hx_aud,
        "orc_evap_area_m2": orc_hx["area_evap_m2"] if orc_hx else None,
        "orc_cond_area_m2": orc_hx["area_cond_m2"] if orc_hx else None,
        "orc_u_w_m2k": orc_u_w_m2k if orc_hx else None,
        "med_gor": med_gor,
        "capacity_factor": capacity_factor,
        "capex_water_aud": capex["water"],
    })
    return result


def _p90_ratio(out_dir, track_c):
    """
    P90 of per-record delivered heat over the design-record delivered heat,
    from Module 3's rating of the Direct-MED exchanger at this track. MED
    output is heat-limited, so MED plant is sized at P90, while the scans
    below work at the design record; this ratio carries the P90 convention
    across. It is about 1 (0.98-1.01) because the station runs at a steady
    load. 1.0 if Module 3 has not been run for the track.
    """
    f = Path(out_dir) / f"gps_module3_output_direct_med_{int(track_c)}.csv"
    if not f.exists():
        return 1.0
    df = pd.read_csv(f)
    q = df["heat_delivered_to_downstream_kw"]
    q_design = float(q[df["Total gas flow kg/s"].idxmax()])
    q_pos = q[q > 0]
    if q_design <= 0 or not len(q_pos):
        return 1.0
    return float(np.percentile(q_pos, MED_SIZING_PERCENTILE)) / q_design


def _med_capacity_factor(out_dir, track_c):
    """
    Average fraction of capacity an MED plant sized at P90 heat actually runs
    at over its operating hours: mean(min(q_row, q_P90)) / q_P90.

    MED output is limited by the heat arriving each hour. Sizing at P90 lets
    the plant use all the heat in 90 % of hours (spilling about 0.3 % of the
    total in the rest), but in a typical hour it receives LESS than its
    design heat and runs part-loaded. RO is unaffected: it runs at capacity
    on grid power whether or not heat arrives.
    """
    f = Path(out_dir) / f"gps_module3_output_direct_med_{int(track_c)}.csv"
    if not f.exists():
        return 1.0
    q = pd.read_csv(f)["heat_delivered_to_downstream_kw"]
    q = q[q > 0]
    if not len(q):
        return 1.0
    p = float(np.percentile(q, MED_SIZING_PERCENTILE))
    return float(np.minimum(q, p).mean() / p) if p > 0 else 1.0


def _med_candidates(heat_kw, t_tbt_c, t_condenser_c):
    """Every physically feasible effect count at this TBT, with GOR and distillate."""
    out = []
    for n in range(1, max_feasible_effects(t_tbt_c, t_condenser_c) + 1):
        gor = gor_for_effects(n)
        out.append((n, gor, distillate_m3_per_day(heat_kw, gor, t_tbt_c, t_condenser_c)))
    return out


def direct_med_design(track_c, data_dir, out_dir, anchor=DESAL_COST_ANCHOR,
                       tbt_floor_c=MED_TBT_FLOOR_C, tbt_step_c=0.5,
                       force_tbt=None, force_n=None):
    """
    Direct-MED with the top brine temperature (TBT) as a design variable,
    coupled to the effect count.

    The loop heats the first effect, where brine boils at TBT. Heat flows
    only hot -> cold, so the loop must stay at least one pinch above TBT
    along the whole exchanger:

        loop_return >= TBT + ORC_EVAPORATOR_PINCH_C   (5 K)

    - the same approach the ORC evaporator is held to, since it is the same
    physical situation: a sensible loop heating an isothermal boiling stream.

    For each TBT the loop return is set AT that floor (a colder return
    shrinks the FGCHE, and the MED side is isothermal, so returning warmer
    gains nothing), the FGCHE is sized, the ID fan cap applied, and every
    feasible effect count up to the physical maximum is costed. The (TBT, N)
    pair with the lowest LCOW wins. The trade it resolves: a higher TBT
    allows more effects (more water per kW) but forces a warmer loop return,
    which enlarges the FGCHE and runs into the fan cap. `force_tbt` and
    `force_n` price one given design instead.

    ASSUMPTION: the loop-to-first-effect heating surface is part of the MED
    plant, costed within the Bhojwani MED correlation (every MED has a
    heat-input section), so no separate exchanger is added. A hot-water first
    effect at a 5 K approach needs more surface than a steam-driven one, so
    this is slightly FAVOURABLE to MED.
    """
    design = _load_design_row_cached(data_dir)
    t_sw, _ = _seawater_rows(data_dir)
    t_cond = condenser_temperature_c(float(np.percentile(t_sw, MED_CONDENSER_DESIGN_PERCENTILE)))

    n_tubes, a_ref = duct_geometry(DUCT_WIDTH_M, PITCH_RATIO)
    gt = gas_side_heat_transfer(design["t_in_c"], design["m_dot_fg"],
                                 design["p_gauge"], design["dust"], a_ref)
    cp_d = dynamic_flue_gas_cp((design["t_in_c"] + track_c) / 2.0, design["mole_fractions"])
    q_target = design["m_dot_fg"] * (N_TRAINS_FITTED / N_GAS_TRAINS) * cp_d * (design["t_in_c"] - track_c)
    q_delivered = q_target * HX_SHELL_EFFICIENCY
    ratio = _p90_ratio(out_dir, track_c)
    cf = _med_capacity_factor(out_dir, track_c)
    q_med = q_delivered * ratio
    t_hot = track_c - LOSS_BUFFER_DELTA_T

    base = {"architecture": "Direct-MED", "track_c": track_c, "t_condenser_c": t_cond,
            "tbt_floor_c": tbt_floor_c, "heat_design_row_kw": q_delivered,
            "p90_ratio": ratio, "capacity_factor": cf, "design_heat_kw": q_med}
    if force_tbt is not None:
        grid = [float(force_tbt)]
    else:
        tbt_lo = max(tbt_floor_c, t_cond + MED_DELTA_T_PER_EFFECT_C)
        tbt_hi = min(MED_TBT_CEILING_C, t_hot - ORC_EVAPORATOR_PINCH_C - tbt_step_c)
        grid = list(np.arange(tbt_lo, tbt_hi + 1e-9, tbt_step_c))

    best, n_sized, n_fan = None, 0, 0
    for tbt in grid:
        tbt = float(tbt)
        lr = tbt + ORC_EVAPORATOR_PINCH_C
        sized = size_fgche(lr, t_hot, q_target, design["t_in_c"],
                            (design["m_dot_fg"] / N_GAS_TRAINS) * cp_d, gt, n_tubes, DUCT_HEIGHT_M)
        if sized is None:
            continue
        n_sized += 1
        rows = sized["rows_316"]
        if _MED_ROW_CAP is not None and rows > _MED_ROW_CAP:
            n_fan += 1
            continue
        fan_kw = gas_side_fan_power(rows, gt)["fan_kw"]       # design record; charged at its annual mean
        pump_kw = loop_pump_power_kw(sized["m_dot_loop_kg_s"])
        for n, gor, water in _med_candidates(q_med, tbt, t_cond):
            if force_n is not None and n != force_n:
                continue
            r = assemble_architecture(
                name="Direct-MED", technology="MED", design_heat_kw=q_med,
                water_m3_per_day=water, hx_area_m2=sized["area_316_m2"],
                fan_kw=fan_kw, loop_pump_kw=pump_kw, orc_gross_kw=0.0,
                process_sec_kwh_m3=MED_PROCESS_SEC_KWH_M3, med_gor=gor, capacity_factor=cf, anchor=anchor)
            if r["lcow_aud_per_m3"] is None:
                continue
            if best is None or r["lcow_aud_per_m3"] < best["record"]["lcow_aud_per_m3"]:
                best = {"record": r, "tbt_c": tbt, "n_effects": n, "gor": gor,
                        "loop_return_c": lr, "rows": rows, "hx_area_m2": sized["area_316_m2"],
                        "fan_kw": fan_kw, "n_max": max_feasible_effects(tbt, t_cond)}
    if best is None:
        reason = ("every TBT above the floor needs more tube rows than the ID fan allows"
                  if n_sized and n_fan == n_sized else "no feasible TBT between floor and ceiling")
        return {**base, "feasible": False, "reason": reason, "record": None}
    rec = best.pop("record")
    rec.update({"track_c": track_c, "tbt_c": best["tbt_c"], "n_effects": best["n_effects"],
                "loop_return_c": best["loop_return_c"]})
    return {**base, **best, "feasible": True, "reason": "", "record": rec,
            "lcow_aud_per_m3": rec["lcow_aud_per_m3"]}


def orc_med_design(track_c, opt, out_dir, data_dir, anchor=DESAL_COST_ANCHOR,
                    tbt_floor_c=MED_TBT_FLOOR_C, force_tc=None, force_path=None, force_n=None):
    """
    ORC+MED with the ORC condensing temperature Tc as the design variable.
    The ORC condenses into the MED's first effect, so

        TBT = Tc - ORC_CONDENSER_PINCH_C

    and both sides are isothermal: there is no temperature-cross risk here,
    only the effect-count cap that Tc sets. Tc is scanned over
    ORC_MED_TC_CANDIDATES_C (the optimiser solves each), with every path and
    every feasible effect count, choosing the lowest LCOW. The ORC condenser
    is costed against brine boiling at TBT, so its LMTD is simply Tc - TBT,
    and the MED is sized at P90 like Direct-MED.
    """
    t_sw, _ = _seawater_rows(data_dir)
    t_cond = condenser_temperature_c(float(np.percentile(t_sw, MED_CONDENSER_DESIGN_PERCENTILE)))
    ratio = _p90_ratio(out_dir, track_c)
    cf = _med_capacity_factor(out_dir, track_c)
    cand = opt[opt["architecture"].str.startswith("orc_med") & (opt["track_c"] == track_c)]
    base = {"architecture": "ORC+MED", "track_c": track_c, "t_condenser_c": t_cond,
            "tbt_floor_c": tbt_floor_c, "p90_ratio": ratio, "capacity_factor": cf}
    best = None
    for _, b in cand.iterrows():
        tc = float(b["tc_c"])
        if force_tc is not None and (abs(tc - force_tc) > 1e-6 or str(b["path"]) != force_path):
            continue
        tbt = tc - ORC_CONDENSER_PINCH_C
        if tbt < tbt_floor_c or tbt > MED_TBT_CEILING_C:
            continue
        q_med = float(b["q_cond_kw"]) * ratio
        ua = (float(b["orc_evap_ua_w_per_k"]) if "orc_evap_ua_w_per_k" in b.index
              and pd.notna(b["orc_evap_ua_w_per_k"]) else None)
        for n, gor, water in _med_candidates(q_med, tbt, t_cond):
            if force_n is not None and n != force_n:
                continue
            r = assemble_architecture(
                name=f"orc_med (path {b['path']})", technology="MED", design_heat_kw=q_med,
                water_m3_per_day=water, hx_area_m2=float(b["area_total_m2"]),
                fan_kw=float(b["fan_kw"]), loop_pump_kw=float(b["loop_pump_kw"]),
                orc_gross_kw=float(b["w_net_kw"]),
                orc_evap_ua_w_per_k=ua, orc_t_cond_c=tc,
                orc_sink_in_c=tbt, orc_sink_out_c=tbt,        # boiling brine: isothermal
                process_sec_kwh_m3=MED_PROCESS_SEC_KWH_M3, med_gor=gor, capacity_factor=cf, anchor=anchor)
            if r["lcow_aud_per_m3"] is None:
                continue
            if best is None or r["lcow_aud_per_m3"] < best["record"]["lcow_aud_per_m3"]:
                best = {"record": r, "tc_c": tc, "path": str(b["path"]), "tbt_c": tbt,
                        "n_effects": n, "gor": gor, "loop_return_c": float(b["loop_return_c"]),
                        "rows": float(b["rows"]), "hx_area_m2": float(b["area_total_m2"]),
                        "fan_kw": float(b["fan_kw"]), "design_heat_kw": q_med,
                        "n_max": max_feasible_effects(tbt, t_cond)}
    if best is None:
        return {**base, "feasible": False,
                "reason": "no fan-feasible ORC design with a TBT inside the floor/ceiling",
                "record": None}
    rec = best.pop("record")
    rec.update({"track_c": track_c, "tbt_c": best["tbt_c"], "n_effects": best["n_effects"],
                "tc_c": best["tc_c"], "loop_return_c": best["loop_return_c"]})
    return {**base, **best, "feasible": True, "reason": "", "record": rec,
            "lcow_aud_per_m3": rec["lcow_aud_per_m3"]}


def direct_ro_design(track_c, ceiling_c, data_dir, u_w_m2k=ORC_U_BASELINE_W_M2K):
    """
    Direct-RO: keep the FGCHE and the waste-heat loop, drop the ORC, and
    preheat the RO feed with ONE loop-to-seawater exchanger instead of the
    ORC's evaporator and condenser.

    This is NOT a no-heat case: the heat-recovery plant, the ID fan duty and
    the loop pump all remain. It isolates the value of the ORC, not the value
    of the heat; the no-heat question is answered by conventional SWRO.

    The loop return is optimised freely, as for the ORC branches, but it is
    SCANNED rather than driven to the floor, because here the trade is
    genuinely two-sided:

        lower loop return -> smaller FGCHE (bigger gas-side LMTD)
                          -> tighter seawater approach, BIGGER preheater

    so an interior optimum is possible and must be found, not assumed. The
    loop return floor is the median seawater intake plus the condenser pinch
    the ORC branches use.
    """
    design = _load_design_row_cached(data_dir)
    t_sw, gas = _seawater_rows(data_dir)
    t_sw_design = float(np.percentile(t_sw, SEAWATER_DESIGN_PERCENTILE))

    n_tubes, a_ref = duct_geometry(DUCT_WIDTH_M, PITCH_RATIO)
    gt = gas_side_heat_transfer(design["t_in_c"], design["m_dot_fg"],
                                 design["p_gauge"], design["dust"], a_ref)
    cp_d = dynamic_flue_gas_cp((design["t_in_c"] + track_c) / 2.0, design["mole_fractions"])
    q_target = design["m_dot_fg"] * (N_TRAINS_FITTED / N_GAS_TRAINS) * cp_d * (design["t_in_c"] - track_c)
    q_delivered = q_target * HX_SHELL_EFFICIENCY
    t_hot = track_c - LOSS_BUFFER_DELTA_T
    if ceiling_c - t_sw_design <= 0 or t_hot <= ceiling_c:
        return None

    feed_kg_s = q_delivered / (SEAWATER_CP_KJ_KGK * (ceiling_c - t_sw_design))

    best = None
    lr_floor = t_sw_design + ORC_CONDENSER_PINCH_C
    for lr in np.arange(lr_floor, t_hot - 5.0, 0.5):
        sized = size_fgche(float(lr), t_hot, q_target, design["t_in_c"],
                            (design["m_dot_fg"] / N_GAS_TRAINS) * cp_d,
                            gt, n_tubes, DUCT_HEIGHT_M)
        if sized is None:
            continue
        rows = sized["rows_316"]
        if _MED_ROW_CAP is not None and rows > _MED_ROW_CAP:
            continue                      # ID fan constraint, same as everywhere
        fan = gas_side_fan_power(rows, gt)

        # loop -> seawater preheater, counterflow, both streams sensible
        dt_hot = t_hot - ceiling_c
        dt_cold = float(lr) - t_sw_design
        if dt_hot <= 0 or dt_cold <= 0:
            continue
        lmtd = dt_hot if abs(dt_hot - dt_cold) < 1e-9 else (dt_hot - dt_cold) / np.log(dt_hot / dt_cold)
        area_pre = q_delivered * 1000.0 / (u_w_m2k * lmtd)

        cand = {
            "loop_return_c": float(lr), "rows": rows,
            "hx_area_m2": sized["area_316_m2"],
            "preheater_area_m2": area_pre,
            "fan_kw": fan["fan_kw"],
            "loop_pump_kw": loop_pump_power_kw(sized["m_dot_loop_kg_s"]),
            "q_delivered_kw": q_delivered, "feed_kg_s": feed_kg_s,
            "capex_exchangers_aud": (fgche_cost_aud(sized["area_316_m2"], N_TRAINS_FITTED)
                                      + installed_cost(hx_purchased_cost_aud(area_pre))),
        }
        # Objective: the annual cost of everything the loop return changes -
        # the exchangers' capital charge and O&M, and the fan and pump
        # electricity. Water production is FIXED for a given track (the feed
        # follows q_delivered and the ceiling, not the loop return), so this
        # is equivalent to minimising LCOW (checked by brute force in
        # validation/independent_checks.py). Capital alone would ignore that
        # fewer rows also cut the fan duty, and the two do not point the same
        # way: fewer rows means a colder loop return, which enlarges the
        # preheater. The heat side runs only while there is flue gas.
        hours = annual_operating_hours("ORC_RO")["preheated_h"]
        crf = capital_recovery_factor(WACC_REAL, ASSET_LIFE_HEAT_RECOVERY_YEARS)
        hs = heat_side_annual_kw(0.0, cand["fan_kw"], cand["loop_pump_kw"])
        cand["annual_cost_aud"] = (cand["capex_exchangers_aud"] * (crf + OANDM_RATE_ORC_HX)
                                    - hs["heat_side_kw"] * hours / 1000.0 * ELECTRICITY_AUD_PER_MWH)
        if best is None or cand["annual_cost_aud"] < best["annual_cost_aud"]:
            best = cand
    return best


def _load_design_row_cached(data_dir, _cache={}):
    """The design record (largest gas flow), loaded once (module5_orc/design_cases.py)."""
    if "row" not in _cache:
        import importlib.util as _il
        sp = _il.spec_from_file_location(
            "rlo", Path(__file__).resolve().parent.parent / "module5_orc" / "design_cases.py")
        mod = _il.module_from_spec(sp); sp.loader.exec_module(mod)
        _cache["row"] = mod.load_design_row(data_dir)
    return _cache["row"]


def run(output_dir, anchor=DESAL_COST_ANCHOR, med_percentile=MED_SIZING_PERCENTILE,
         loop_objective="lcow"):
    """
    Every architecture at every track, written to
    outputs/gps_module6_lcow_<cost basis>_<loop objective>.csv.

    `loop_objective` selects the optimiser output to consume: "lcow", the
    loop return at its minimum feasible value, which is the LCOW optimum.
    """
    out_dir = Path(output_dir)
    opt = pd.read_csv(out_dir / f"loop_optimisation_summary_{loop_objective}.csv")
    # the measured duct is the reporting baseline; the other duct sizes are
    # sensitivity cases
    opt = opt[opt["duct"].str.contains("MEASURED")]

    # HARD ID fan constraint (common/fan_margin.py). Designs whose bank is
    # deeper than the fan can carry on its worst operating record are not
    # candidates at all - the model's only genuine feasibility limit, as
    # opposed to a cost the objective can trade away.
    if "fan_feasible" in opt.columns:
        n_before = len(opt)
        opt = opt[opt["fan_feasible"].astype(bool)]
        print(f"  ID fan constraint: {n_before - len(opt)} of {n_before} "
              f"designs excluded, {len(opt)} remain")

    records = []

    data_dir = str(Path(out_dir).parent / "data")
    med_designs = []

    for track in sorted(opt["track_c"].unique()):
        # --- Direct-MED: TBT and effect count co-optimised ---
        dmed = direct_med_design(float(track), data_dir, out_dir, anchor=anchor)
        med_designs.append({k: v for k, v in dmed.items() if k != "record"})
        if dmed["feasible"]:
            records.append(dmed["record"])
        else:
            print(f"  Direct-MED @ {int(track)}C: no feasible design - {dmed['reason']}")

        # --- ORC+RO: take the optimised design point per path ---
        for arch in ("orc_ro_current", "orc_ro_advanced"):
            sel = opt[(opt["architecture"] == arch) & (opt["track_c"] == track)]
            if sel.empty:
                continue
            # the path (A subcritical or B transcritical) with the most net
            # electricity, on the annual-mean basis the LCOW charges
            annual = [heat_side_annual_kw(w, f, p)["heat_side_kw"] for w, f, p in
                      zip(sel["w_net_kw"], sel["fan_kw"], sel["loop_pump_kw"])]
            best = sel.iloc[int(np.argmax(annual))]

            if arch.startswith("orc_ro"):
                ceiling = 38.0 if "current" in arch else 50.0
                perf = ro_annual_performance(float(best["q_cond_kw"]), ceiling,
                                              str(Path(out_dir).parent / "data"))
                if perf is None:
                    continue
                sec_pre = perf["sec_preheated_kwh_m3"]
                sec_a = perf["sec_ambient_kwh_m3"]
                water_pd = perf["water_m3_per_day"]
                tech = "SWRO"

            records.append(assemble_architecture(
                orc_evap_ua_w_per_k=float(best["orc_evap_ua_w_per_k"])
                    if "orc_evap_ua_w_per_k" in best.index
                    and pd.notna(best["orc_evap_ua_w_per_k"]) else None,
                orc_t_cond_c=float(best["tc_c"]),
                orc_sink_in_c=perf["design_seawater_c"],
                orc_sink_out_c=ceiling,
                name=f"{arch} (path {best['path']})", technology=tech,
                design_heat_kw=float(best["q_cond_kw"]),
                water_m3_per_day=water_pd,
                hx_area_m2=float(best["area_total_m2"]),
                fan_kw=float(best["fan_kw"]), loop_pump_kw=float(best["loop_pump_kw"]),
                orc_gross_kw=float(best["w_net_kw"]),
                process_sec_kwh_m3=sec_pre, sec_ambient_kwh_m3=sec_a,
                anchor=anchor))
            # carry the optimised loop return through to the output, so the
            # reported design is traceable without re-reading the optimiser
            records[-1]["loop_return_c"] = float(best["loop_return_c"])

        # --- ORC+MED: condensing temperature and effect count co-optimised ---
        omed = orc_med_design(float(track), opt, out_dir, data_dir, anchor=anchor)
        med_designs.append({k: v for k, v in omed.items() if k != "record"})
        if omed["feasible"]:
            records.append(omed["record"])

        # --- Direct-RO: keep the heat, drop the ORC ---
        dro = direct_ro_design(float(track), 38.0, str(Path(out_dir).parent / "data"))
        if dro is not None:
            perf = ro_annual_performance(dro["q_delivered_kw"], 38.0,
                                          str(Path(out_dir).parent / "data"))
            if perf is not None:
                records.append(assemble_architecture(
                    name="Direct-RO (no ORC)", technology="SWRO",
                    design_heat_kw=dro["q_delivered_kw"],
                    water_m3_per_day=perf["water_m3_per_day"],
                    hx_area_m2=dro["hx_area_m2"],
                    fan_kw=dro["fan_kw"], loop_pump_kw=dro["loop_pump_kw"],
                    orc_gross_kw=0.0,
                    process_sec_kwh_m3=perf["sec_preheated_kwh_m3"],
                    sec_ambient_kwh_m3=perf["sec_ambient_kwh_m3"],
                    extra_heat_capex_aud=installed_cost(
                        hx_purchased_cost_aud(dro["preheater_area_m2"])),
                    anchor=anchor))
                records[-1]["loop_return_c"] = dro["loop_return_c"]
                records[-1]["preheater_area_m2"] = dro["preheater_area_m2"]

        for r in records:
            r.setdefault("track_c", track)

    # --- Conventional SWRO: the NO-HEAT reference ---
    # Answers the question Direct-RO cannot: is the waste heat worth
    # recovering AT ALL? Direct-RO still carries the whole heat-recovery
    # plant and the full ID fan duty, so it only isolates the ORC's value.
    # Capacity is HELD at the largest RO plant's, so the comparison isolates
    # the heat rather than confounding it with scale.
    ro_rows = [r for r in records if r["technology"] == "SWRO" and r["water_m3_per_day"] > 0]
    if ro_rows:
        held = max(r["water_m3_per_day"] for r in ro_rows)
        t_sw, _ = _seawater_rows(str(Path(out_dir).parent / "data"))
        sec_amb = float(np.mean([ro_specific_energy(t, r=RO_RECOVERY)["sec_kwh_m3"] for t in t_sw]))
        records.append(assemble_architecture(
            name="Conventional SWRO (no heat recovery)", technology="SWRO",
            design_heat_kw=0.0, water_m3_per_day=held,
            hx_area_m2=0.0, fan_kw=0.0, loop_pump_kw=0.0, orc_gross_kw=0.0,
            process_sec_kwh_m3=sec_amb, anchor=anchor))
        records[-1]["track_c"] = None

    if med_designs:
        pd.DataFrame(med_designs).to_csv(out_dir / "med_design_summary.csv", index=False)

    df_out = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)}
                            for r in records])
    path = out_dir / f"gps_module6_lcow_{anchor}_{loop_objective}.csv"
    df_out.to_csv(path, index=False)
    print(f"wrote {path} ({len(df_out)} rows)")
    return df_out


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    df = run(str(root / "outputs"))
    cols = ["architecture", "track_c", "water_m3_per_day", "operating_hours",
            "net_electricity_kw", "lcow_aud_per_m3", "capital_component_aud_per_m3",
            "opex_component_aud_per_m3"]
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(df[cols].round(3).to_string(index=False))
