"""
Discretised T-Q (composite curve) pinch check, applied to every ORC path
(A, B and C). Splits the heat exchange into equal-duty segments, computes
the LOCAL temperature of both streams at each segment boundary, and checks
the local approach temperature everywhere, not just at the two end points.
A simple inlet/outlet check misses internal crossings: a stream with an
isothermal boiling plateau (or any non-linear T-h relationship) can look
fine at both ends while crossing the other stream in the middle. An
endpoint-only check once passed a subcritical evaporator with a -27.3 K
internal crossing.

Composite curve convention: define a shared cumulative-duty coordinate
Q in [0, Q_total], with Q=0 at the HOT END of a counterflow exchanger
(hot stream at its inlet, cold stream at its outlet - the two hottest
points meet) and Q=Q_total at the COLD END (hot stream at its outlet,
cold stream at its inlet - the two coldest points meet). Both streams'
local enthalpy (and hence temperature, via CoolProp) are evaluated at
each shared Q, which is the standard way to check feasibility without
needing real exchanger geometry.

PHASE-CHANGE POINTS. An evenly spaced grid alone cannot certify a boiling
stream. Its temperature has a KINK where boiling starts (the bubble point)
and another where it ends (the dew point), and the closest approach sits at
one of those kinks. With 20 even segments, the bubble point of every
subcritical ORC evaporator falls between the last two check points (at
about 98.5 % of duty, between 95 % and 100 %), and an even-grid check reads
5.00 K where the true approach is 3.88-4.27 K. So the check also evaluates
both streams at their own saturation points, located exactly from the
fluid's saturation enthalpies at the stream's pressure. Supercritical
streams have no such points, and the pressurised loop water never reaches
saturation in this exchanger.
"""

import numpy as np
import CoolProp.CoolProp as CP


def _phase_change_fractions(fluid, p_pa, h_at_hot_end, h_at_cold_end):
    """
    Duty fractions (0 = hot end, 1 = cold end) at which a stream crosses its
    saturated-liquid (bubble) or saturated-vapour (dew) enthalpy at p_pa.
    Only interior points are returned - the two ends are always checked.
    Empty for a supercritical stream, and for one that never reaches
    saturation inside this exchanger.
    """
    try:
        if p_pa >= CP.PropsSI("Pcrit", fluid):
            return []
        span = h_at_hot_end - h_at_cold_end
        if abs(span) < 1e-12:
            return []
        out = []
        for quality in (0.0, 1.0):
            f = (h_at_hot_end - CP.PropsSI("H", "P", p_pa, "Q", quality, fluid)) / span
            if 1e-9 < f < 1.0 - 1e-9:
                out.append(float(f))
        return out
    except ValueError:        # CoolProp cannot resolve saturation for this state
        return []


def discretized_pinch_check(hot_fluid, p_hot_pa, h_hot_in, h_hot_out, m_dot_hot,
                             cold_fluid, p_cold_pa, h_cold_in, h_cold_out, m_dot_cold,
                             n_segments=20, min_pinch_k=5.0, include_phase_change_points=True):
    """
    Checks local approach temperature at n_segments+1 evenly spaced points
    along the shared duty coordinate, plus - by default - exactly at each
    stream's bubble and dew points, where a boiling stream's kinks are.
    Both pressures are assumed constant across their own stream (standard
    HX assumption, consistent with the rest of this model).

    include_phase_change_points=False gives the evenly spaced check alone.
    It is kept for validation (validation/pinch_check_tests.py), which shows
    what it misses, not for design.

    Returns dict: valid (bool, True iff min approach >= min_pinch_k
    everywhere), min_pinch_k (the worst-case local approach found),
    pinch_location_fraction (0=hot end, 1=cold end), q_total_w and
    q_total_kw,
    energy_balance_check_pct (should be ~0, a self-consistency check
    that the two streams' declared duties actually match), the phase-change
    fractions that were added, and the full profile for inspection and
    plotting. Profile points are NOT evenly spaced when phase-change points
    are present, so anything integrating along the profile must use each
    point's duty_fraction (common/costing.area_from_pinch_profile_m2 does).
    """
    q_total_hot = m_dot_hot * (h_hot_in - h_hot_out)
    q_total_cold = m_dot_cold * (h_cold_out - h_cold_in)
    energy_balance_check_pct = abs(q_total_hot - q_total_cold) / max(abs(q_total_hot), 1e-9) * 100.0

    q_total = q_total_hot  # reported duty (hot-side basis, matches Module 3's convention)

    # Each stream is walked across its OWN total duty at the shared
    # FRACTIONAL position, rather than both being walked across the hot
    # stream's absolute duty. The two duties are deliberately not equal
    # here - the loop gives up q_total_hot but the working fluid only
    # absorbs q_total_cold, the difference being the HX shell loss to
    # ambient (HX_SHELL_EFFICIENCY, ~1.5%). Using the hot duty for both
    # would stretch the cold stream ~1.5% too far, so that it appears to
    # start BELOW its own inlet state, overstating the cold-end approach
    # by ~0.7K. That is small in general but sits exactly where the
    # transcritical path's binding constraint is, so it would flatter
    # Path B specifically. Distributing the shell loss
    # proportionally along the exchanger is the standard assumption and
    # keeps both streams spanning exactly their own end states.
    fractions = {i / n_segments for i in range(n_segments + 1)}
    phase = []
    if include_phase_change_points:
        phase = (_phase_change_fractions(hot_fluid, p_hot_pa, h_hot_in, h_hot_out)
                 + _phase_change_fractions(cold_fluid, p_cold_pa, h_cold_out, h_cold_in))
        fractions |= set(phase)
    phase_set = set(phase)

    profile = []
    min_pinch = float("inf")
    pinch_fraction = 0.0

    for fraction in sorted(fractions):
        h_hot_local = h_hot_in - fraction * q_total_hot / m_dot_hot
        h_cold_local = h_cold_out - fraction * q_total_cold / m_dot_cold

        t_hot_local = CP.PropsSI("T", "P", p_hot_pa, "H", h_hot_local, hot_fluid) - 273.15
        t_cold_local = CP.PropsSI("T", "P", p_cold_pa, "H", h_cold_local, cold_fluid) - 273.15

        approach = t_hot_local - t_cold_local
        profile.append({"duty_fraction": fraction, "t_hot_c": t_hot_local, "t_cold_c": t_cold_local,
                        "approach_k": approach, "phase_change_point": fraction in phase_set})

        if approach < min_pinch:
            min_pinch = approach
            pinch_fraction = fraction

    return {
        "valid": min_pinch >= min_pinch_k,
        "min_pinch_k": min_pinch,
        "pinch_location_fraction": pinch_fraction,
        # m_dot [kg/s] x dh [J/kg] is in WATTS. Both units are published,
        # each named for what it contains, so a caller cannot take one for
        # the other (a 1000x error).
        "q_total_w": q_total,
        "q_total_kw": q_total / 1000.0,
        "energy_balance_check_pct": energy_balance_check_pct,
        "phase_change_fractions": sorted(phase),
        "profile": profile,
    }
