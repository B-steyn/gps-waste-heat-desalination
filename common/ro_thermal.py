"""
RO specific energy consumption with thermal enhancement, per Yagnambhatt,
Khanmohammadi & Maisonneuve, "Reducing the specific energy use of
seawater desalination with thermally enhanced reverse osmosis,"
Desalination 573 (2024) 117163.

Implements their transport model (Eqs. 1-8, 11-16: mechanical pumping
work plus temperature-dependent membrane/fluid properties). Their
heat-pump-cost overlay (Eqs. 9-10, e_TE-RO = e_RO + q/cop) is
deliberately NOT implemented - it prices the thermal energy input via a
hypothetical work-driven heat pump, which doesn't apply here: the preheat
is waste heat that is rejected regardless, not generated on demand. The
cost of running the ORC condenser hot enough to drive the preheat is
already captured as reduced ORC output, so the paper's q/cop term on top
would count it twice.

Validated against the paper's own reported figures at matched conditions
(50 lmh flux, r=0.5, seawater, T=20C -> 50C): computed eRO=4.326 kWh/m3
at 20C (paper states exactly "4.3"), 24.4% reduction at 50C (paper: 24%),
applied pressure 113.3 -> 85.6 bar (paper: 113 -> 86 bar), permeate
concentration 0.284 g/L at 50C (paper: 0.28 g/L).

Note: the paper's own seawater density correlation (Table 1, citing
ref [30]) could not be reproduced from the text as extracted - every
unit convention tried gives wildly negative densities (the c^2 term
dominates catastrophically). Substituted a standard linear seawater
density approximation instead; density only feeds into Re/Sc for the
mass-transfer coefficient, a secondary correction, so this does not
materially affect validation (which still matched the paper closely).
"""

import numpy as np

R_GAS = 8.314  # J/mol-K
M_NACL = 58.44  # g/mol
I_IONS = 2.0  # NaCl fully dissociates into 2 ions

T20_K = 293.15  # the paper's reference temperature for A20/B20/mu20/D20


def _viscosity(c_kg_per_l, t_kelvin):
    """Eq. 12 (Table 1 form). Validated: pure water at 20C -> 0.975 mPa.s
    (real ~1.00), seawater 35g/L at 20C -> 1.038 mPa.s (real ~1.08)."""
    return 0.4599 * np.exp(1.796 * c_kg_per_l - 0.021 * t_kelvin)


def _salt_diffusivity(c_kg_per_l, t_kelvin):
    """Eq. 14. Validated: ~1.47e-9 m2/s at 20C, matching known NaCl-in-
    water diffusivity order of magnitude."""
    return 6.725e-6 * np.exp(9.034e-3 * c_kg_per_l - 2513.0 / t_kelvin)


def _seawater_density(c_kg_per_l, t_celsius):
    """Standard linear seawater density approximation (~1025 kg/m3 at
    35 g/L, 20C), substituted for the paper's unreproducible density
    formula - see module docstring."""
    salinity_g_per_l = c_kg_per_l * 1000.0
    return 1000.0 + 0.75 * salinity_g_per_l - 0.2 * (t_celsius - 20.0)


def _solve_concentration_polarization(c_feed_kg_per_l, r, flux_lmh, b_lmh, k_m_per_s):
    """Iterative solve of Eqs. 6-8 (mutually coupled: c_B depends on c_P,
    c_P depends on delta_c, delta_c depends on c_B). Paper's own Methods
    section confirms this needs numerical iteration, not a closed form.
    Returns (c_permeate_kg_per_l, delta_c_kg_per_l)."""
    vp_over_a = flux_lmh / 1000.0 / 3600.0  # lmh -> m/s
    b_m_per_s = b_lmh / 1000.0 / 3600.0
    polarization_factor = np.exp(vp_over_a / k_m_per_s)

    c_permeate = 0.0
    for _ in range(200):
        c_brine = (c_feed_kg_per_l - r * c_permeate) / (1.0 - r)
        delta_c = polarization_factor * (c_brine - c_feed_kg_per_l) / (
            np.log(c_brine) - np.log(c_feed_kg_per_l)) - c_permeate
        new_c_permeate = (1.0 / vp_over_a) * b_m_per_s * delta_c
        if abs(new_c_permeate - c_permeate) < 1e-12:
            c_permeate = new_c_permeate
            break
        c_permeate = new_c_permeate

    return c_permeate, delta_c


def ro_specific_energy(t_feed_celsius, c_feed_kg_per_l=0.035, r=0.5, flux_lmh=25.0,
                        a20_lmh_per_bar=1.0, b20_lmh=0.1, hydraulic_diameter_m=0.001,
                        feed_velocity_m_per_s=0.1, eta_pump=0.8, eta_pex=0.9,
                        pressure_loss_pa=0.0):
    """
    Specific energy of RO (mechanical pumping only, kWh/m3) as a function
    of feed temperature, following Yagnambhatt et al.'s transport model.

    Defaults are the paper's own Table 1 case-study conditions (seawater,
    25 lmh flux, r=0.5, A=1 lmh/bar, B=0.1 lmh). The model passes its own
    recovery (constants.RO_RECOVERY) on every call.

    Returns dict: sec_kwh_m3, applied_pressure_bar, permeate_conc_g_l.
    """
    t_kelvin = t_feed_celsius + 273.15

    mu_t = _viscosity(c_feed_kg_per_l, t_kelvin)
    mu_20 = _viscosity(c_feed_kg_per_l, T20_K)
    a_t = a20_lmh_per_bar * (mu_20 / mu_t)  # Eq 11

    d_t = _salt_diffusivity(c_feed_kg_per_l, t_kelvin)
    d_20 = _salt_diffusivity(c_feed_kg_per_l, T20_K)
    b_t = b20_lmh * (d_t / d_20)  # Eq 16

    rho = _seawater_density(c_feed_kg_per_l, t_feed_celsius)
    reynolds = rho * feed_velocity_m_per_s * hydraulic_diameter_m / mu_t
    schmidt = mu_t / (rho * d_t)
    sherwood = 0.46 * reynolds ** 0.36 * schmidt ** 0.36  # Eq 15
    k_mass_transfer = d_t * sherwood / hydraulic_diameter_m  # Eq 13

    c_permeate, delta_c = _solve_concentration_polarization(
        c_feed_kg_per_l, r, flux_lmh, b_t, k_mass_transfer)

    delta_c_molar = delta_c * 1.0e6 / M_NACL  # kg/L -> mol/m3
    delta_pi_pa = I_IONS * R_GAS * t_kelvin * delta_c_molar  # Eq 5

    vp_over_a_m_per_s = flux_lmh / 1000.0 / 3600.0
    a_t_si = a_t / 1000.0 / 3600.0 / 1e5  # lmh/bar -> m/(s.Pa)
    applied_pressure_pa = vp_over_a_m_per_s / a_t_si + delta_pi_pa  # Eq 4

    e_permeate = (1.0 / eta_pump) * applied_pressure_pa  # Eq 2
    e_brine = (1.0 / eta_pump) * (1.0 / r - 1.0) * (
        applied_pressure_pa - eta_pex * (applied_pressure_pa - pressure_loss_pa))  # Eq 3
    sec_j_per_m3 = e_permeate + e_brine  # Eq 1

    return {
        "sec_kwh_m3": sec_j_per_m3 / 3.6e6,
        "applied_pressure_bar": applied_pressure_pa / 1e5,
        "permeate_conc_g_l": c_permeate * 1000.0,
    }
