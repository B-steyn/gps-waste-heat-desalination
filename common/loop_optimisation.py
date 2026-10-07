"""
Chooses the intermediate loop's return temperature.

The trade-off being resolved, all at a FIXED loop duty (the gas side
sets how much heat comes out of the flue gas; the loop only chooses how
to carry it):

  raising loop_return_c
    -> narrows the loop dT, so RAISES loop mass flow
    -> FLATTENS the loop's T-Q curve, so the working fluid can evaporate
       hotter before the pinch binds            [more ORC output]
    -> but shrinks the FGCHE's LMTD, so needs more area / more tube rows
       -> more gas-side pressure drop           [more ID fan power]
    -> and moves more water                     [more loop pump power]

On net electricity alone the ID fan term creates an interior optimum. The
fan power scales roughly with 1/duct_area^2, so `gas_transport` and the
duct geometry are passed in as arguments rather than read from constants,
and the optimisation can be re-run for any duct.

The objective is pluggable. Net electricity (ORC output - ID fan - loop
pump) sees the fan's operating cost but not the extra exchanger area's
capital, and the two pull the same way, so the LCOW optimum sits at or
below the net-electricity optimum. The reported designs use the LCOW
optimum, found by find_minimum_feasible_loop_return.
"""

import numpy as np

from common.fgche_sizing import size_fgche, gas_side_fan_power, loop_pump_power_kw
from common.constants import P_WATER_LOOP_PA


def net_electricity_objective(entry):
    """
    MAXIMISE net electricity: the right objective for the thermodynamic
    question "what design produces the most power?".

    It is not the economic objective. It sees the ID fan's operating cost
    but not the exchanger's capital, and it ignores that the ORC competes
    with the desalination plant for the same heat. Scored on LCOW, the
    optimum moves about 12 K lower.
    """
    return entry["w_net_kw"] - entry["fan_kw"] - entry["loop_pump_kw"]


def lcow_objective_factory(lcow_evaluator):
    """
    MINIMISE LCOW - the economic objective, and the default for
    reported results.

    Takes a callable that turns one `evaluate_loop_return` entry into an
    LCOW figure, so this module stays free of any dependency on the
    costing layer (which would otherwise be circular: costing needs the
    design, the design needs the objective, the objective needs costing).
    The caller injects the economics.

    Returned as a MAXIMISED score (negative LCOW) so `optimise_loop_return`
    keeps a single convention - it always maximises.
    """
    def _score(entry):
        lcow = lcow_evaluator(entry)
        if lcow is None:
            return -float("inf")
        return -lcow
    return _score


def evaluate_loop_return(loop_return_c, path_solver, t_water_hot_c, q_target_kw,
                          t_gas_in_c, c_gas_per_duct_kw_per_k, gas_transport,
                          n_tubes_per_row, duct_height_m, u_key="u_316",
                          p_loop_pa=P_WATER_LOOP_PA):
    """
    Sizes the FGCHE for one candidate loop_return_c and solves the ORC
    path against the resulting loop, returning every term needed to
    score it. Returns None if the candidate is infeasible (no lift, or
    no pinch-valid ORC operating point).

    Per-duct vs total convention follows Module 3 exactly: q_target_kw
    and the returned area are TOTAL (both ducts); the UA solve and
    gas_transport are PER DUCT.
    """
    sized = size_fgche(loop_return_c, t_water_hot_c, q_target_kw, t_gas_in_c,
                        c_gas_per_duct_kw_per_k, gas_transport, n_tubes_per_row,
                        duct_height_m, p_loop_pa=p_loop_pa)
    if sized is None:
        return None

    # the alloy selector picks which sized area/row count this scoring
    # pass uses; 316L is the reference case throughout the model
    area_total_m2 = sized["area_316_m2"] if u_key == "u_316" else sized["area_904_m2"]
    rows = sized["rows_316"] if u_key == "u_316" else sized["rows_904"]

    fan = gas_side_fan_power(rows, gas_transport)
    loop_pump_kw = loop_pump_power_kw(sized["m_dot_loop_kg_s"])

    result = path_solver(sized["h_loop_hot"], sized["h_loop_return"], sized["m_dot_loop_kg_s"])
    constrained = result.get("constrained")
    if constrained is None:
        return None

    return {
        "loop_return_c": loop_return_c,
        "m_dot_loop_kg_s": sized["m_dot_loop_kg_s"],
        "ua_per_duct_kw_per_k": sized["ua_per_duct_kw_per_k"],
        "area_total_m2": area_total_m2,
        "rows": rows,
        "dp_pa": fan["dp_pa"],
        "fan_kw": fan["fan_kw"],
        "loop_pump_kw": loop_pump_kw,
        "w_net_kw": constrained["cycle"]["w_net_kw"],
        "q_cond_kw": constrained["cycle"]["q_cond_kw"],
        "cycle": constrained["cycle"],
        "pinch": constrained["pinch"],
        "operating_point_c": constrained.get("t_evap_c", constrained.get("t3_c")),
        "result": result,
    }


def optimise_loop_return(path_solver, t_water_hot_c, q_target_kw, t_gas_in_c, t_gas_out_c,
                          c_gas_per_duct_kw_per_k, gas_transport, n_tubes_per_row, duct_height_m,
                          lr_min_c=None, lr_max_c=None, max_dp_pa=None, objective=net_electricity_objective,
                          coarse_step_c=2.0, refine_step_c=0.25, u_key="u_316",
                          gas_side_approach_c=5.0, p_loop_pa=P_WATER_LOOP_PA):
    """
    Coarse scan then local refinement. Cheap because the response is
    smooth in loop_return_c apart from the discrete tube-row steps in
    the fan term (rows are integer, so fan cost is a step function and
    the optimum tends to sit just below a row increment).

    Bounds:
      lower - defaults to just above the condenser temperature; the real
        binding lower limit is the path's own cold-end pinch
        (loop_return_c >= T2 + pinch), which is enforced naturally
        by the path solver returning no valid point below it.
      upper - defaults to the gas-side approach limit: the loop water
        must stay below the gas OUTLET temperature by a workable
        approach, otherwise the FGCHE pinches out.
      max_dp_pa - optional hard cap representing the existing ID fan's
        static-pressure margin. Exceeding it in reality forces a fan
        replacement or booster, a discrete CAPEX step this smooth
        objective cannot see, so it is imposed as a constraint rather
        than costed. Leave None to leave it unbounded and inspect the
        scan.

    Returns dict: best (the winning entry, None if nothing feasible),
    scan (every feasible entry evaluated, for plotting the trade-off),
    and rejected_for_dp (entries excluded only by max_dp_pa, so the cost
    of that constraint is visible rather than silent).
    """
    if lr_max_c is None:
        lr_max_c = t_gas_out_c - gas_side_approach_c
    if lr_min_c is None:
        lr_min_c = 30.0
    lr_max_c = min(lr_max_c, t_water_hot_c - 1.0)
    if lr_min_c >= lr_max_c:
        return {"best": None, "scan": [], "rejected_for_dp": []}

    def sweep(values):
        out = []
        for lr in values:
            entry = evaluate_loop_return(
                lr, path_solver, t_water_hot_c, q_target_kw, t_gas_in_c,
                c_gas_per_duct_kw_per_k, gas_transport, n_tubes_per_row,
                duct_height_m, u_key=u_key, p_loop_pa=p_loop_pa)
            if entry is not None:
                out.append(entry)
        return out

    coarse = sweep(np.arange(lr_min_c, lr_max_c + 1e-9, coarse_step_c))
    if not coarse:
        return {"best": None, "scan": [], "rejected_for_dp": []}

    feasible = [e for e in coarse if max_dp_pa is None or e["dp_pa"] <= max_dp_pa]
    rejected = [e for e in coarse if max_dp_pa is not None and e["dp_pa"] > max_dp_pa]
    if not feasible:
        return {"best": None, "scan": coarse, "rejected_for_dp": rejected}

    anchor = max(feasible, key=objective)["loop_return_c"]
    fine = sweep(np.arange(max(lr_min_c, anchor - coarse_step_c),
                            min(lr_max_c, anchor + coarse_step_c) + 1e-9, refine_step_c))

    all_entries = {round(e["loop_return_c"], 4): e for e in coarse}
    all_entries.update({round(e["loop_return_c"], 4): e for e in fine})
    scan = [all_entries[k] for k in sorted(all_entries)]

    feasible = [e for e in scan if max_dp_pa is None or e["dp_pa"] <= max_dp_pa]
    rejected = [e for e in scan if max_dp_pa is not None and e["dp_pa"] > max_dp_pa]
    best = max(feasible, key=objective) if feasible else None

    return {"best": best, "scan": scan, "rejected_for_dp": rejected}


def find_minimum_feasible_loop_return(path_solver, t_water_hot_c, q_target_kw, t_gas_in_c,
                                       c_gas_per_duct_kw_per_k, gas_transport,
                                       n_tubes_per_row, duct_height_m,
                                       lr_lower_bound_c=30.0, tol_c=0.1,
                                       u_key="u_316", p_loop_pa=P_WATER_LOOP_PA,
                                       gas_side_approach_c=5.0, t_gas_out_c=None):
    """
    The LCOW-optimal loop return, obtained as the MINIMUM FEASIBLE value.

    This is a shortcut, and it is only legitimate because the floor
    hypothesis was verified: LCOW rises monotonically with loop_return_c
    in every duct, architecture and track tested (12 of 12 cases, including
    the largest duct, where a cheaper fan could plausibly have broken it;
    re-checked by the loop-return scans in analysis/supporting_calculations,
    CR5 and CR15). Given monotonicity, the optimum is the boundary, so a
    bisection against the feasibility constraint replaces a scan that would
    otherwise need the whole cost layer evaluated at every candidate.

    IF THAT HYPOTHESIS IS EVER INVALIDATED - a different working fluid, a
    materially different cost structure, a much larger electricity price -
    this function must be replaced by `optimise_loop_return` with
    `lcow_objective_factory`. The scan machinery is retained for exactly
    that reason.

    Feasibility is set by the path's own cold-end pinch: below some
    loop_return_c the solver returns no pinch-valid operating point.
    Bisects for that boundary.

    Returns the evaluated entry at the minimum feasible loop return, or
    None if nothing is feasible anywhere in range.
    """
    upper = t_water_hot_c - 1.0
    if t_gas_out_c is not None:
        upper = min(upper, t_gas_out_c - gas_side_approach_c)

    def _eval(lr):
        return evaluate_loop_return(lr, path_solver, t_water_hot_c, q_target_kw, t_gas_in_c,
                                     c_gas_per_duct_kw_per_k, gas_transport, n_tubes_per_row,
                                     duct_height_m, u_key=u_key, p_loop_pa=p_loop_pa)

    # confirm the upper end is feasible at all before bisecting downward
    top = _eval(upper)
    if top is None:
        # walk down coarsely to find any feasible point
        found = None
        for lr in np.arange(upper, lr_lower_bound_c, -2.0):
            top = _eval(float(lr))
            if top is not None:
                found = float(lr)
                break
        if found is None:
            return None
        upper = found

    lo, hi = lr_lower_bound_c, upper
    best = top
    while hi - lo > tol_c:
        mid = (lo + hi) / 2.0
        e = _eval(mid)
        if e is None:
            lo = mid          # infeasible -> the boundary is above mid
        else:
            best = e
            hi = mid          # feasible -> push the boundary down
    return best
