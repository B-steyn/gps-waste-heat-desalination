"""
epsilon-NTU heat exchanger design and rating (Kays & London; Incropera,
Fundamentals of Heat and Mass Transfer).

Distinguishes DESIGN (given a target outlet condition, find the area
needed) from RATING (given a fixed, already-built exchanger, find what it
does at off-design conditions). The exchanger is sized ONCE at the design
point, which fixes its area and hence its UA at reference conditions, and
that fixed geometry is then rated at every other operating record with
that record's own flow conditions. Sizing every record separately would
amount to a piece of hardware renegotiating its own area at every
timestep.

Tube arrangement: BOTH FLUIDS UNMIXED. Considered treating the gas side
as "mixed" (a common simplification for a single wide, thin crossflow
exchanger, e.g. a car radiator's air side) since the duct is an open
cross-section rather than physically divided gas channels - but this
model's tube bank is DEEP (tens of rows, per rows_316/rows_904's typical
range of areas), and for a multi-row in-line/staggered bank, lateral
gas mixing doesn't meaningfully re-homogenize between every single row
- local wakes and flow lanes persist. The
alternative "one fluid mixed" correlation has two directionally
different forms (Cmax-mixed vs Cmin-mixed) that could not be confidently
distinguished from memory alone without risking a backwards assignment
- given the numerical difference between all three crossflow variants is
modest at this model's NTU/Cr range (checked: a few percentage points of
effectiveness), "both unmixed" was preferred as the better-grounded,
unambiguous choice rather than guess at a directional formula.

Design method: an LMTD with an F-correction factor (Bowman/TEMA charts)
would also serve for the one-time design-point sizing step. Its
closed-form F-correction has enough transcription risk (multiple
near-identical variants exist depending on which R/P convention a given
source uses) that it isn't used here. Instead, the SAME verified
epsilon-NTU relationship is used for design too (inverting it: given a
target duty at the design row, bisect on UA until the predicted q
matches) - this is standard, equally valid practice (e.g. Cengel's Heat
Transfer teaches epsilon-NTU as a full alternative to LMTD+F for design,
not just for rating), and avoids the F-correction formula and its
transcription risk entirely by using one consistent, already-validated
method throughout.
"""

import numpy as np


def effectiveness_crossflow_both_unmixed(ntu, c_r):
    """
    Kays & London / Incropera correlation for a single-pass crossflow HX
    with both fluids unmixed. Verified: reduces to the universal Cr=0
    limit (eps = 1 - exp(-NTU), the condenser/evaporator/Cmin->infinite
    limit) as c_r -> 0.
    """
    if c_r <= 1e-9:
        return 1.0 - np.exp(-ntu)
    return 1.0 - np.exp((1.0 / c_r) * (ntu ** 0.22) * (np.exp(-c_r * (ntu ** 0.78)) - 1.0))


def rate_fixed_hx(ua_kw_per_k, c_gas_kw_per_k, c_water_kw_per_k, t_gas_in_c, t_water_in_c):
    """
    Rates a FIXED-UA heat exchanger at the given (row-specific) flow
    conditions. UA itself should already reflect the current row's
    actual gas-side film coefficient (area is fixed, but U is not - U
    depends on the flow-rate-dependent gas-side h, so the caller
    recomputes UA per row from the fixed area and the row's own
    Reynolds/Nusselt numbers, not just scales a single reference UA).

    Returns dict: q_actual_kw, t_gas_out_c, t_water_out_c, effectiveness,
    ntu, c_r.
    """
    c_min = min(c_gas_kw_per_k, c_water_kw_per_k)
    c_max = max(c_gas_kw_per_k, c_water_kw_per_k)
    c_r = c_min / c_max if c_max > 0 else 0.0

    if c_min <= 0 or ua_kw_per_k <= 0:
        return {"q_actual_kw": 0.0, "t_gas_out_c": t_gas_in_c, "t_water_out_c": t_water_in_c,
                "effectiveness": 0.0, "ntu": 0.0, "c_r": c_r}

    ntu = ua_kw_per_k / c_min
    eps = effectiveness_crossflow_both_unmixed(ntu, c_r)

    q_max_kw = c_min * (t_gas_in_c - t_water_in_c)
    q_actual_kw = eps * q_max_kw

    t_gas_out_c = t_gas_in_c - q_actual_kw / c_gas_kw_per_k
    t_water_out_c = t_water_in_c + q_actual_kw / c_water_kw_per_k

    return {
        "q_actual_kw": q_actual_kw,
        "t_gas_out_c": t_gas_out_c,
        "t_water_out_c": t_water_out_c,
        "effectiveness": eps,
        "ntu": ntu,
        "c_r": c_r,
    }


def solve_required_ua(q_target_kw, c_gas_kw_per_k, c_water_kw_per_k, t_gas_in_c, t_water_in_c, max_ntu=50.0):
    """
    Inverse epsilon-NTU: finds the UA (kW/K) needed so that
    rate_fixed_hx(...) reproduces q_target_kw at the given design-point
    flow/temperature conditions. Used once, at the chosen design row, to
    fix the HX's UA (and from that, via the existing area/U relationship,
    its physical area and tube-row count) - the design-method
    counterpart to LMTD+F, using the same verified effectiveness
    correlation instead. Bisects on NTU (monotonic in effectiveness, so
    well-posed) then converts back to UA.
    """
    c_min = min(c_gas_kw_per_k, c_water_kw_per_k)
    c_max = max(c_gas_kw_per_k, c_water_kw_per_k)
    c_r = c_min / c_max if c_max > 0 else 0.0

    q_max_kw = c_min * (t_gas_in_c - t_water_in_c)
    if q_max_kw <= 0:
        return 0.0
    eps_target = q_target_kw / q_max_kw
    eps_target = min(eps_target, 0.999)  # eps=1 needs infinite NTU

    lo, hi = 0.0, max_ntu
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if effectiveness_crossflow_both_unmixed(mid, c_r) < eps_target:
            lo = mid
        else:
            hi = mid

    ntu_required = (lo + hi) / 2.0
    return ntu_required * c_min
