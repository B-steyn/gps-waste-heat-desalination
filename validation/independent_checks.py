"""
Independent verification of the model's defining calculations, on the
thermodynamic side (combustion, gas properties, the exchanger, the ORC cycles,
RO and MED) and the economic side (hours, discounting, the LCOW build-up and
the desalination capital), re-checking the chain the headline rests on.

Each check RECOMPUTES a result from first principles and its own property
calls. It reuses the model's inputs (data, constants, chosen designs) but not
its calculation functions, and compares the recomputed value with what the
model reported. A check is:
  PASS  the model matches the independent calculation
  FLAG  a finding: the model's number is right for its own assumption, but
        the assumption is questionable. Each flag states its size
  FAIL  the model does not reproduce its own physics or economics

The last run_all.py step. Writes validation/independent_checks.log; exits
non-zero on any FAIL.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module5_orc"))

import common.constants as C
import common.economic_constants as ec

OUT = ROOT / "outputs"
LOG = []
TALLY = {"PASS": 0, "FLAG": 0, "FAIL": 0}


def say(line=""):
    print(line, flush=True)
    LOG.append(line)


def check(name, ok, detail, flag=False):
    status = "FLAG" if flag else ("PASS" if ok else "FAIL")
    TALLY[status] += 1
    say(f"  [{status}] {name}: {detail}")


def close(a, b, rel):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


# --------------------------------------------------------------------------
# inputs: the design row, the chosen designs, Module 6's results
# --------------------------------------------------------------------------
g = pd.read_csv(OUT / "gps_module6_lcow_bhojwani_global_lcow.csv")
opt = pd.read_csv(OUT / "loop_optimisation_summary_lcow.csv")
opt = opt[opt.duct.str.contains("MEASURED")]
m1 = pd.read_csv(OUT / "GPS_Module1_Output.csv")

from design_cases import load_design_row          # noqa: E402  (inputs only)
D = load_design_row(str(ROOT / "data"))
TRACK = 85.0
row = lambda arch, t=TRACK: g[(g.architecture == arch) & ((g.track_c == t) if t else g.track_c.isna())].iloc[0]
ORC = row("orc_ro_current (path A)")
DRO = row("Direct-RO (no ORC)")
PLAIN = g[g.architecture.str.startswith("Conventional SWRO")].iloc[0]
DES = opt[(opt.architecture == "orc_ro_current") & (opt.track_c == TRACK) & (opt.path == "A")].iloc[0]

# ==========================================================================
say("T1  Combustion and flue-gas composition (Module 1)")
# ==========================================================================
# Re-derived from the coal analysis, independently of common/coal_combustion.
f_dry = 1 - C.MOIST_AD / 100
ash_ar = (C.ASH_AD / 100) / f_dry * (1 - C.TOTAL_MOIST_AR)
daf = 1 - C.TOTAL_MOIST_AR - ash_ar
c, h, o, n, s = (x * daf for x in (C.C_DAF, C.H_DAF, C.O_DAF, C.N_DAF, C.S_DAF))
mol = {"CO2": c / 12.011, "H2O": h / 2.016 + C.TOTAL_MOIST_AR / 18.015, "Nf": n / 28.013}
so2_0 = s / 32.06
so3_f = so2_0 * C.CONVERSION_SO2_TO_SO3
mol["SO3"] = so3_f * (1 - C.ASH_CAPTURE_EFF)
mol["SO2"] = so2_0 - so3_f
o2_theo = mol["CO2"] + h / 4.032 + so2_0 + 0.5 * so3_f - o / 31.998
gas = pd.read_csv(ROOT / "data" / "GPS_Operating_Data_Clean.csv")
gas.columns = gas.columns.str.strip()
o2 = (gas["A side flue gas o2 %"] + gas["B side flue gas o2 %"]) / 2 / 100
ok_rows = gas["Total gas flow kg/s"] > 1.0
# wet O2: y = (lam-1) n_theo / n_tot, with n_tot = (n_prod - n_theo) + lam n_theo / 0.21
# (air taken as 21 % O2, the rest N2), solved for lam
n_prod = mol["CO2"] + mol["H2O"] + mol["SO2"] + mol["SO3"] + mol["Nf"]
lam = (o2_theo + o2 * (n_prod - o2_theo)) / (o2_theo * (1 - o2 / 0.21))
n_tot = n_prod + (lam - 1) * o2_theo + lam * o2_theo * 0.79 / 0.21
y_h2o = mol["H2O"] / n_tot
y_o2_back = (lam - 1) * o2_theo / n_tot
check("exhaust O2 closes on the measured O2", np.allclose(y_o2_back, o2, atol=1e-12),
      f"max residual {np.max(np.abs(y_o2_back - o2)):.1e} over {len(o2)} rows")
mm = m1.set_index(m1.Timestamp.astype(str).str.strip())
gg = gas.assign(k=gas.Timestamp.astype(str).str.strip()).set_index("k")
common_k = mm.index.intersection(gg.index)
lam_m = mm.loc[common_k, "dynamic_lambda"].values
lam_i = pd.Series(lam.values, index=gg.index).loc[common_k].values
yh_m = mm.loc[common_k, "flue_gas_y_h2o"].values
yh_i = pd.Series(y_h2o.values, index=gg.index).loc[common_k].values
good = np.isfinite(lam_m)
check("lambda and H2O fraction reproduce Module 1",
      np.allclose(lam_m[good], lam_i[good], rtol=1e-9) and np.allclose(yh_m[good], yh_i[good], rtol=1e-9),
      f"{good.sum()} rows; lambda {np.nanmin(lam_m):.3f}-{np.nanmax(lam_m):.3f}, "
      f"y_H2O {np.nanmin(yh_m):.4f}-{np.nanmax(yh_m):.4f}")
# combustion-air humidity: the balance takes all H2O from the fuel
for w_kg in (0.010, 0.015):
    n_air = lam * o2_theo / 0.21
    extra = n_air * w_kg * 28.965 / 18.015
    yh_hum = (mol["H2O"] + extra) / (n_tot + extra)
    tdp = lambda y: CP.PropsSI("T", "P", float(y) * 101325.0, "Q", 0, "Water") - 273.15
    d_dp = tdp(np.nanmean(yh_hum[ok_rows])) - tdp(np.nanmean(y_h2o[ok_rows]))
    check(f"combustion-air humidity (omega {w_kg:.3f} kg/kg)", True,
          f"not in the balance: it would raise mean y_H2O {np.nanmean(y_h2o[ok_rows]):.4f} -> "
          f"{np.nanmean(yh_hum[ok_rows]):.4f}, the water dew point by {d_dp:+.1f} K and the "
          f"acid dew point by {27.6 * np.log10(np.nanmean(yh_hum[ok_rows]) / np.nanmean(y_h2o[ok_rows])):+.1f} K",
          flag=True)

# ==========================================================================
say("\nT2  Dew points")
# ==========================================================================
p_tot = (C.P_AMBIENT_KPA + (gas["Fabric Filter Outlet Pressure (kPa) A side"]
                             + gas["Fabric Filter Outlet Pressure (kPa) B side"]) / 2) / 101.325
y_so3 = mol["SO3"] / n_tot
ph2o, pso3 = y_h2o * p_tot, y_so3 * p_tot
# Okkes (1987), partial pressures in atm
adp = 203.25 + 27.6 * np.log10(ph2o) + 10.83 * np.log10(pso3) + 1.06 * (np.log10(pso3) + 8) ** 2.19
adp_m = mm.loc[common_k, "calculated_adp_c"].values
adp_i = pd.Series(adp.values, index=gg.index).loc[common_k].values
ok = np.isfinite(adp_m)
check("acid dew point (Okkes) reproduces Module 1", np.allclose(adp_m[ok], adp_i[ok], atol=1e-9),
      f"mean {np.mean(adp_i[ok]):.1f} C, range {np.min(adp_i[ok]):.1f}-{np.max(adp_i[ok]):.1f} C")
wdp = np.array([CP.PropsSI("T", "P", p * 101325.0, "Q", 0, "Water") - 273.15
                for p in ph2o[ok_rows]])
check("water dew point from the H2O partial pressure", 20 < wdp.mean() < 50,
      f"mean {wdp.mean():.1f} C, range {wdp.min():.1f}-{wdp.max():.1f} C ")

# ==========================================================================
say("\nT3  Heat available from the flue gas (design row, 85 C track)")
# ==========================================================================
y = D["mole_fractions"]
spc = {"y_co2": ("CarbonDioxide", 44.0095), "y_h2o": ("Water", 18.01528),
       "y_o2": ("Oxygen", 31.9988), "y_n2": ("Nitrogen", 28.0134)}
ytot = sum(y[k] for k in spc)
m_mix = sum(y[k] / ytot * spc[k][1] for k in spc)
w = {k: y[k] / ytot * spc[k][1] / m_mix for k in spc}


def cp0(t_c):
    return sum(w[k] * CP.PropsSI("Cp0mass", "T", t_c + 273.15, "P", 101325.0, spc[k][0]) for k in spc) / 1000


t_in = float(D["t_in_c"])
ts = np.linspace(TRACK, t_in, 401)
dh_exact = np.trapezoid([cp0(t) for t in ts], ts)          # integral of cp dT, kJ/kg
q_exact = D["m_dot_fg"] * C.N_TRAINS_FITTED / C.N_GAS_TRAINS * dh_exact
cp_mean = cp0((t_in + TRACK) / 2)
q_meanT = D["m_dot_fg"] * C.N_TRAINS_FITTED / C.N_GAS_TRAINS * cp_mean * (t_in - TRACK)
check("mean-temperature cp x dT against the integral of cp dT", close(q_meanT, q_exact, 1e-3),
      f"{q_meanT:,.1f} vs {q_exact:,.1f} kW ({100 * (q_meanT / q_exact - 1):+.3f} %); "
      f"gas {D['m_dot_fg']:.1f} kg/s from {t_in:.2f} C to {TRACK:.0f} C")
check("heat delivered to the loop = gas duty x shell efficiency (Direct-RO design heat)",
      close(q_meanT * C.HX_SHELL_EFFICIENCY, float(DRO.design_heat_kw), 1e-6),
      f"{q_meanT * C.HX_SHELL_EFFICIENCY:,.1f} kW vs model {float(DRO.design_heat_kw):,.1f} kW")
r_mix = 8314.462 / m_mix
check("gas density basis", True,
      f"the model uses dry air's R = 287.05 J/kg-K; the flue gas (M = {m_mix:.2f} g/mol) has "
      f"R = {r_mix:.1f}, so it is {287.05 / r_mix - 1:.1%} denser than modelled. Fan power per row "
      f"scales with 1/rho^2: {100 * ((r_mix / 287.05) ** 2 - 1):+.1f} %", flag=True)

# ==========================================================================
say("\nT4  FGCHE design (ORC+RO path A, 85 C track, measured duct)")
# ==========================================================================
lr, t_hot = float(DES.loop_return_c), TRACK - C.LOSS_BUFFER_DELTA_T
p_loop = C.P_WATER_LOOP_PA
h_hot = CP.PropsSI("H", "T", t_hot + 273.15, "P", p_loop, "Water")
h_ret = CP.PropsSI("H", "T", lr + 273.15, "P", p_loop, "Water")
m_loop = q_meanT / ((h_hot - h_ret) / 1000)
check("loop water mass flow from the energy balance", close(m_loop, float(DES.m_dot_loop_kg_s), 1e-6),
      f"{m_loop:.2f} vs model {float(DES.m_dot_loop_kg_s):.2f} kg/s (liquid water at 10 bar, {lr:.2f}-{t_hot:.0f} C)")
# gas-side U, recomputed from Zukauskas
s_t = C.PITCH_RATIO * C.TUBE_DO_M
n_tubes = int(np.floor(C.DUCT_WIDTH_M / s_t))
a_min = C.DUCT_HEIGHT_M * (C.DUCT_WIDTH_M - n_tubes * C.TUBE_DO_M)
p_gas = (C.P_AMBIENT_KPA + D["p_gauge"]) * 1000
rho = p_gas / (287.05 * (t_in + 273.15))
v_max = D["m_dot_fg"] / C.N_GAS_TRAINS / (rho * a_min)
re = rho * v_max * C.TUBE_DO_M / C.MU_GAS_FLUE
h_gas = C.ZUKAUSKAS_C * re ** C.ZUKAUSKAS_M * C.PRANDTL_GAS ** C.ZUKAUSKAS_N * C.K_GAS_FLUE / C.TUBE_DO_M
r_foul = C.FOULING_RESISTANCE_BASELINE * (D["dust"] / C.FOULING_DUST_NORMALIZER)
u = 1 / (1 / h_gas + 1 / 4000.0 + r_foul + C.TUBE_WALL_M / C.K_316L)
check("Zukauskas range", 1e3 <= re <= 2e5,
      f"Re_max = {re:,.0f} (valid 10^3-2x10^5 for C = 0.35, m = 0.6); h_gas {h_gas:.1f}, U {u:.1f} W/m2-K")
# UA: the crossflow design must need MORE UA than ideal counterflow at the same duty
c_g = D["m_dot_fg"] / C.N_GAS_TRAINS * cp_mean
c_w = q_meanT / C.N_TRAINS_FITTED / (t_hot - lr)
dt1, dt2 = t_in - t_hot, TRACK - lr
lmtd = (dt1 - dt2) / np.log(dt1 / dt2)
ua_cf = q_meanT / C.N_TRAINS_FITTED / lmtd
area_model = float(DES.area_total_m2)
ua_model = area_model / C.N_TRAINS_FITTED * u / 1000
f_corr = ua_cf / ua_model
check("FGCHE area: crossflow UA >= counterflow UA, with F in a sane range", 0.75 <= f_corr <= 1.0,
      f"counterflow UA {ua_cf:,.1f} kW/K per duct; model {ua_model:,.1f} -> implied F = {f_corr:.3f}; "
      f"area {area_model:,.0f} m2 for {q_meanT:,.0f} kW")
rows = np.ceil(area_model / C.N_TRAINS_FITTED / (n_tubes * np.pi * C.TUBE_DO_M * C.DUCT_HEIGHT_M))
dp = rows * C.TUBE_ROW_DP_FACTOR * 0.5 * rho * v_max ** 2
v_duct = D["m_dot_fg"] / rho / C.N_GAS_TRAINS
fan = C.N_TRAINS_FITTED * v_duct * dp / C.FAN_EFFICIENCY / 1000
check("tube rows and ID fan power", rows == DES.rows and close(fan, float(ORC.fan_kw), 1e-6),
      f"{rows:.0f} rows, dP {dp:.0f} Pa, fan {fan:.1f} kW vs model {float(ORC.fan_kw):.1f} kW")
pump = m_loop * C.PIPE_LENGTH_M * C.PIPE_FRICTION_PAM / C.RHO_LIQUID_WATER / C.PUMP_EFFICIENCY / 1000
check("loop pump (friction only, closed loop)", close(pump, float(ORC.loop_pump_kw), 1e-6),
      f"{pump:.2f} kW vs model {float(ORC.loop_pump_kw):.2f} kW")
# second law across the FGCHE: gas entropy falls, water entropy rises more
s_gas = D["m_dot_fg"] * np.trapezoid([cp0(t) / (t + 273.15) for t in ts], ts)   # kW/K given up
s_w = m_loop * (CP.PropsSI("S", "T", t_hot + 273.15, "P", p_loop, "Water")
                - CP.PropsSI("S", "T", lr + 273.15, "P", p_loop, "Water")) / 1000
t0 = C.T_AMBIENT_CELSIUS + 273.15
check("FGCHE entropy generation positive", s_w - s_gas > 0,
      f"S_gen = {s_w - s_gas:.3f} kW/K, exergy destroyed {t0 * (s_w - s_gas):,.0f} kW")

# ==========================================================================
say("\nT5  ORC cycle (path A, 85 C track)")
# ==========================================================================
F = "R1234ze(E)"
te, tc = float(DES.operating_point_c), float(DES.tc_c)
q_in = q_meanT * C.HX_SHELL_EFFICIENCY
pc_, pe_ = (CP.PropsSI("P", "T", t + 273.15, "Q", 0, F) for t in (tc, te))
h1, s1 = (CP.PropsSI(k, "P", pc_, "Q", 0, F) for k in ("H", "S"))
h2 = h1 + (CP.PropsSI("H", "P", pe_, "S", s1, F) - h1) / C.ORC_PUMP_ISENTROPIC_EFF
s2 = CP.PropsSI("S", "P", pe_, "H", h2, F)
t3 = te + C.ORC_SUPERHEAT_C
h3, s3 = (CP.PropsSI(k, "T", t3 + 273.15, "P", pe_, F) for k in ("H", "S"))
h4 = h3 - C.ORC_TURBINE_ISENTROPIC_EFF * (h3 - CP.PropsSI("H", "P", pc_, "S", s3, F))
s4 = CP.PropsSI("S", "P", pc_, "H", h4, F)
m_wf = q_in / ((h3 - h2) / 1000)
w_t, w_p = m_wf * (h3 - h4) / 1000, m_wf * (h2 - h1) / 1000
w_net = C.ORC_GENERATOR_EFF * w_t - w_p - C.ORC_PARASITIC_FRACTION_OF_GROSS * w_t
q_out = m_wf * (h4 - h1) / 1000
check("net output reproduces the model", close(w_net, float(DES.w_net_kw), 1e-6),
      f"{w_net:.2f} kW vs model {float(DES.w_net_kw):.2f} kW (T_evap {te:.2f} C, T_cond {tc:.0f} C, "
      f"{m_wf:.1f} kg/s of {F})")
check("first law: Q_in + W_pump = W_turbine + Q_out", close(q_in + w_p, w_t + q_out, 1e-9),
      f"{q_in + w_p:,.2f} = {w_t + q_out:,.2f} kW; Q_out {q_out:,.1f} vs model condenser {float(DES.q_cond_kw):,.1f}")
check("condenser heat reproduces the model", close(q_out, float(DES.q_cond_kw), 1e-6),
      f"{q_out:,.2f} kW")
eta = (w_t - w_p) / q_in
carnot = 1 - (tc + 273.15) / (te + 273.15)
check("efficiency below Carnot", 0 < eta < carnot,
      f"{eta:.3%} against Carnot {carnot:.3%} between T_evap and T_cond ({eta / carnot:.0%} of it)")
t4 = CP.PropsSI("T", "P", pc_, "H", h4, F) - 273.15
t4_sat = tc
check("turbine exhaust is dry vapour", t4 > t4_sat + 0.01,
      f"{t4:.2f} C at the condenser pressure, {t4 - t4_sat:.2f} K above saturation")
sg_pump, sg_turb = m_wf * (s2 - s1) / 1000, m_wf * (s4 - s3) / 1000
check("entropy generation in pump and turbine >= 0", sg_pump >= 0 and sg_turb >= 0,
      f"pump {sg_pump * 1000:.2f} W/K, turbine {sg_turb * 1000:.1f} W/K")
s_loop = m_loop * (CP.PropsSI("S", "T", lr + 273.15, "P", p_loop, "Water")
                 - CP.PropsSI("S", "T", t_hot + 273.15, "P", p_loop, "Water")) / 1000 * (q_in / q_meanT)
sg_evap = s_loop + m_wf * (s3 - s2) / 1000
check("entropy generation in the evaporator >= 0", sg_evap > 0, f"{sg_evap:.3f} kW/K")
_w = pd.read_csv(ROOT / "data" / "GPS_Water_related_data.csv").rename(columns=str.strip)
_g = pd.read_csv(ROOT / "data" / "GPS_Operating_Data_Clean.csv").rename(columns=str.strip)
_w["k"], _g["k"] = _w.Timestamp.astype(str).str.strip(), _g.Timestamp.astype(str).str.strip()
t_sw = float(np.percentile(_g[_g["Total gas flow kg/s"] > 1.0].merge(_w[["k", "Sea water intake temp"]], on="k")
                           ["Sea water intake temp"], 50))     # operating rows, as Module 6
m_sw = q_out / (4.0 * (C.RO_CEILING_CURRENT_MEMBRANE_C - t_sw))
sg_cond = m_sw * 4.0 * np.log((C.RO_CEILING_CURRENT_MEMBRANE_C + 273.15) / (t_sw + 273.15)) - m_wf * (s4 - s1) / 1000
check("entropy generation in the condenser >= 0", sg_cond > 0,
      f"{sg_cond:.3f} kW/K (seawater {t_sw:.1f} -> {C.RO_CEILING_CURRENT_MEMBRANE_C:.0f} C)")
# pinch: an independent 4,001-point composite plus the exact bubble point
hb = CP.PropsSI("H", "P", pe_, "Q", 0, F)
fr = np.unique(np.r_[np.linspace(0, 1, 4001), (h3 - hb) / (h3 - h2)])
h_cold = h3 - fr * (h3 - h2)
t_cold = np.array([CP.PropsSI("T", "P", pe_, "H", x, F) for x in h_cold]) - 273.15
# distributed shell loss, as the model: the hot stream spans h_hot -> h_ret
h_w = h_hot - fr * (h_hot - h_ret)
t_w = np.array([CP.PropsSI("T", "P", p_loop, "H", x, "Water") for x in h_w]) - 273.15
pinch = float(np.min(t_w - t_cold))
check("evaporator minimum approach >= 5 K (independent 4,001-point check)", pinch >= 5.0 - 5e-3,
      f"{pinch:.3f} K at duty fraction {fr[np.argmin(t_w - t_cold)]:.4f} from the hot end")
h_w_lump = (h_hot - (q_meanT - q_in) / m_loop * 1000) - fr * (q_in / m_loop * 1000)
t_w_lump = np.array([CP.PropsSI("T", "P", p_loop, "H", x, "Water") for x in h_w_lump]) - 273.15
pl = float(np.min(t_w_lump - t_cold))
check("pinch if the 1.5 % loop loss sat in the supply pipe instead", True,
      f"{pl:.3f} K; the model spreads the loss along the exchanger", flag=pl < 5.0 - 5e-3)
pr = pe_ / pc_
check("expander pressure ratio", True,
      f"{pr:.3f} ({pe_ / 1e5:.2f} / {pc_ / 1e5:.2f} bar), far below any ORC expander's design range",
      flag=True)

# ==========================================================================
say("\nT5b Transcritical ORC (path B, 90 C track, where Module 6 reports it)")
# ==========================================================================
DB = opt[(opt.architecture == "orc_ro_current") & (opt.track_c == 90.0) & (opt.path == "B")].iloc[0]
cp90 = cp0((t_in + 90.0) / 2)
q90 = D["m_dot_fg"] * C.N_TRAINS_FITTED / C.N_GAS_TRAINS * cp90 * (t_in - 90.0)
qin_b = q90 * C.HX_SHELL_EFFICIENCY
p_hi = 1.05 * CP.PropsSI("Pcrit", F)
b1h, b1s = h1, s1
b2h = b1h + (CP.PropsSI("H", "P", p_hi, "S", b1s, F) - b1h) / C.ORC_PUMP_ISENTROPIC_EFF
t3b = float(DB.operating_point_c)
b3h, b3s = (CP.PropsSI(k, "T", t3b + 273.15, "P", p_hi, F) for k in ("H", "S"))
b4h = b3h - C.ORC_TURBINE_ISENTROPIC_EFF * (b3h - CP.PropsSI("H", "P", pc_, "S", b3s, F))
mb = qin_b / ((b3h - b2h) / 1000)
wtb, wpb = mb * (b3h - b4h) / 1000, mb * (b2h - b1h) / 1000
wnb = C.ORC_GENERATOR_EFF * wtb - wpb - C.ORC_PARASITIC_FRACTION_OF_GROSS * wtb
check("path B net output and condenser heat reproduce the model",
      close(wnb, float(DB.w_net_kw), 1e-6) and close(mb * (b4h - b1h) / 1000, float(DB.q_cond_kw), 1e-6),
      f"{wnb:.2f} kW vs {float(DB.w_net_kw):.2f}; T3 {t3b:.2f} C at {p_hi / 1e5:.1f} bar "
      f"(1.05 x Pcrit), pressure ratio {p_hi / pc_:.2f}")
x4b = CP.PropsSI("Q", "P", pc_, "H", b4h, F)
tcrit = CP.PropsSI("Tcrit", F) - 273.15
check("path B efficiency below Carnot", (wtb - wpb) / qin_b < 1 - (tc + 273.15) / (t3b + 273.15),
      f"{(wtb - wpb) / qin_b:.2%} against Carnot {1 - (tc + 273.15) / (t3b + 273.15):.2%} at T3")
# T3 below the critical temperature at 1.05 x Pcrit is a dense, liquid-like state, so the expansion
# ends two-phase. path_c_tfc.py's own rule: never score a two-phase expansion with a dry-vapour
# efficiency (0.80); its conservative two-phase value is 0.60.
b4h_2ph = b3h - 0.60 * (b3h - CP.PropsSI("H", "P", pc_, "S", b3s, F))
w2 = mb * (b3h - b4h_2ph) / 1000
wn2 = C.ORC_GENERATOR_EFF * w2 - wpb - C.ORC_PARASITIC_FRACTION_OF_GROSS * w2
check("path B expansion: dry-vapour turbine efficiency applied to a wet expansion", True,
      f"T3 {t3b:.1f} C is {tcrit - t3b:.0f} K below Tcrit ({CP.PropsSI('D', 'T', t3b + 273.15, 'P', p_hi, F):.0f} kg/m3, "
      f"liquid-like), so the exhaust is two-phase at quality {x4b:.2f}. Scored at path C's two-phase 0.60 instead of "
      f"0.80, net output is {wn2:+.0f} kW, not {wnb:+.0f} kW", flag=True)
hh90 = CP.PropsSI("H", "T", 85.0 + 273.15, "P", p_loop, "Water")
hr90 = CP.PropsSI("H", "T", float(DB.loop_return_c) + 273.15, "P", p_loop, "Water")
frb = np.linspace(0, 1, 2001)
tcb = np.array([CP.PropsSI("T", "P", p_hi, "H", x, F) for x in b3h - frb * (b3h - b2h)]) - 273.15
twb = np.array([CP.PropsSI("T", "P", p_loop, "H", x, "Water") for x in hh90 - frb * (hh90 - hr90)]) - 273.15
check("path B heater minimum approach >= 5 K (2,001 points; supercritical, no plateau)",
      np.min(twb - tcb) >= 5.0 - 5e-3, f"{np.min(twb - tcb):.3f} K vs model {float(DB.min_pinch_k):.3f} K")

# ==========================================================================
say("\nT5c ORC+MED (reported design)")
# ==========================================================================
om = g[g.architecture.str.startswith("orc_med")].sort_values("lcow_aud_per_m3").iloc[0]
check("ORC condenser feeds the first effect across a 5 K isothermal approach",
      close(float(om.tc_c) - float(om.tbt_c), C.ORC_CONDENSER_PINCH_C, 1e-9),
      f"Tc {float(om.tc_c):.0f} C -> TBT {float(om.tbt_c):.0f} C, {int(om.n_effects)} effects, "
      f"{float(om.track_c):.0f} C track")
gor_om = sum(0.95 ** k for k in range(1, int(om.n_effects) + 1))
lat = 2499.5698 - 2.204864 * (0.5 * (om.tbt_c + 34.7)) - 0.001596 * (0.5 * (om.tbt_c + 34.7)) ** 2
check("ORC+MED distillate = GOR x Q / latent heat (condenser at the warmest hour, 34.7 C)",
      close(gor_om * float(om.design_heat_kw) / lat * 86.4, float(om.water_m3_per_day), 2e-3),
      f"{gor_om * float(om.design_heat_kw) / lat * 86.4:,.0f} vs model {float(om.water_m3_per_day):,.0f} m3/day")

# ==========================================================================
say("\nT8  Direct-RO loop-to-seawater preheater (85 C track)")
# ==========================================================================
lr_d = float(DRO.loop_return_c)
dth = t_hot - C.RO_CEILING_CURRENT_MEMBRANE_C
_tdes = float(np.percentile(pd.read_csv(ROOT / "data" / "GPS_Operating_Data_Clean.csv").rename(columns=str.strip)
                            .assign(k=lambda d: d.Timestamp.astype(str).str.strip())
                            .query("`Total gas flow kg/s` > 1.0")
                            .merge(pd.read_csv(ROOT / "data" / "GPS_Water_related_data.csv").rename(columns=str.strip)
                                   .assign(k=lambda d: d.Timestamp.astype(str).str.strip())[["k", "Sea water intake temp"]],
                                   on="k")["Sea water intake temp"], 50))
dtc = lr_d - _tdes
lm = (dth - dtc) / np.log(dth / dtc)
area_pre = float(DRO.design_heat_kw) * 1000 / (1000.0 * lm)
check("preheater approaches and area (counterflow, U = 1,000 W/m2-K)",
      min(dth, dtc) >= C.ORC_CONDENSER_PINCH_C and close(area_pre, float(DRO.preheater_area_m2), 1e-6),
      f"hot end {dth:.1f} K, cold end {dtc:.1f} K, LMTD {lm:.1f} K -> {area_pre:,.0f} m2 "
      f"vs model {float(DRO.preheater_area_m2):,.0f} m2")

# ==========================================================================
say("\nT6  RO feed, water and SEC")
# ==========================================================================
from common.ro_thermal import ro_specific_energy    # noqa: E402  (validated against its source 2026-09-14)
wat = pd.read_csv(ROOT / "data" / "GPS_Water_related_data.csv").rename(columns=str.strip)
sw_rows = pd.read_csv(ROOT / "data" / "GPS_Operating_Data_Clean.csv").rename(columns=str.strip)
sw_rows["k"] = sw_rows.Timestamp.astype(str).str.strip()
wat["k"] = wat.Timestamp.astype(str).str.strip()
mrg = sw_rows[sw_rows["Total gas flow kg/s"] > 1.0].merge(wat[["k", "Sea water intake temp"]], on="k")
t_design = float(np.percentile(mrg["Sea water intake temp"], 50))
for lab, r in (("Direct-RO", DRO), ("ORC+RO", ORC)):
    feed = float(r.design_heat_kw) / (C.SEAWATER_CP_KJ_KGK * (C.RO_CEILING_CURRENT_MEMBRANE_C - t_design))
    water = feed * C.RO_RECOVERY * 86.4
    check(f"{lab} water from its heat", close(water, float(r.water_m3_per_day), 1e-6),
          f"{float(r.design_heat_kw):,.0f} kW heats {feed:.1f} kg/s from {t_design:.2f} to 38 C -> "
          f"{water:,.0f} m3/day at {C.RO_RECOVERY:.0%} recovery")
cp_sw = CP.PropsSI("C", "T", (t_design + 38) / 2 + 273.15, "P", 101325, "INCOMP::MITSW[0.035]") / 1000
check("seawater specific heat against Sharqawy et al. (2010)", close(C.SEAWATER_CP_KJ_KGK, cp_sw, 2e-3),
      f"model {C.SEAWATER_CP_KJ_KGK:.4f}, MIT seawater {cp_sw:.4f} kJ/kg-K at 35 g/kg and "
      f"{(t_design + 38) / 2:.0f} C (a typical typed 3.93 would be 1.9 % low)")
lo, hi = ro_specific_energy(t_design, r=C.RO_RECOVERY), ro_specific_energy(38.0, r=C.RO_RECOVERY)
# ideal minimum work at recovery r (van 't Hoff, 35 g/kg NaCl-equivalent)
pi_f = 2 * 35 / 58.44 * 8.314 * (t_design + 273.15) / 100      # bar
w_min = pi_f * 1e5 * np.log(1 / (1 - C.RO_RECOVERY)) / C.RO_RECOVERY / 3.6e6
check("SEC above the thermodynamic minimum, and falling with temperature",
      lo["sec_kwh_m3"] > w_min and hi["sec_kwh_m3"] < lo["sec_kwh_m3"],
      f"{lo['sec_kwh_m3']:.3f} -> {hi['sec_kwh_m3']:.3f} kWh/m3 (-{1 - hi['sec_kwh_m3'] / lo['sec_kwh_m3']:.1%}); "
      f"minimum {w_min:.2f} kWh/m3; applied {lo['applied_pressure_bar']:.1f} bar against a concentrate "
      f"osmotic pressure of about {pi_f / (1 - C.RO_RECOVERY):.0f} bar")

# ==========================================================================
say("\nT7  MED (Direct-MED, reported design)")
# ==========================================================================
med = g[g.architecture == "Direct-MED"].sort_values("lcow_aud_per_m3").iloc[0]
e = 0.95
gor = sum(e ** k for k in range(1, int(med.n_effects) + 1))
check("GOR from the effect cascade, and GOR < N", close(gor, float(med.med_gor), 1e-9) and gor < med.n_effects,
      f"{int(med.n_effects)} effects, GOR {gor:.3f} vs model {float(med.med_gor):.3f}")
check("first effect below the loop return (no temperature cross)",
      float(med.loop_return_c) >= float(med.tbt_c) + 5 - 1e-6,
      f"TBT {float(med.tbt_c):.1f} C, loop return {float(med.loop_return_c):.1f} C, loop supply "
      f"{float(med.track_c) - C.LOSS_BUFFER_DELTA_T:.0f} C")

# ==========================================================================
say("\nE1  Operating hours, discount rate, CRF")
# ==========================================================================
hrs_med, hrs_ro = 8760 * ec.GPS_AVAILABILITY, 8760 * ec.DESAL_OWN_AVAILABILITY
check("operating hours", close(float(med.operating_hours), hrs_med, 1e-9)
      and close(float(DRO.operating_hours), hrs_ro, 1e-9) and close(float(DRO.preheated_hours), hrs_med, 1e-9),
      f"MED {hrs_med:,.0f} h; RO {hrs_ro:,.0f} h, of which {hrs_med:,.0f} with heat")
i = (1 + ec.WACC_NOMINAL) / (1 + ec.INFLATION) - 1
crf = i * (1 + i) ** 25 / ((1 + i) ** 25 - 1)
check("real discount rate (Fisher) and 25-year CRF", close(i, float(DRO.real_discount_rate), 1e-12),
      f"{i:.4%}, CRF {crf:.5f}")

# ==========================================================================
say("\nE2a Heat-side energy over the operating records")
# ==========================================================================
# Recomputed from the raw records: for a fixed bank, added fan power is
# rows x K x (1/2 rho v_max^2) x V = const x m^3 / rho^2, so its annual mean is
# mean(m^3/rho^2) / (m^3/rho^2 at the design record). The ORC's heat follows
# gas flow, so its output scales with mean(m) / m_design.
raw = pd.read_csv(ROOT / "data" / "GPS_Operating_Data_Clean.csv")
raw.columns = raw.columns.str.strip()
raw = raw[raw["Total gas flow kg/s"] > 1.0]
m_r = raw["Total gas flow kg/s"].to_numpy()
p_r = ((raw["Fabric Filter Outlet Pressure (kPa) A side"] + raw["Fabric Filter Outlet Pressure (kPa) B side"]) / 2).to_numpy()
rho_r = (C.P_AMBIENT_KPA + p_r) * 1000 / (287.05 * (raw["FF_Outlet_Temp_Avg_C"].to_numpy() + 273.15))
k = int(np.argmax(m_r))
f_fan = float(np.mean(m_r ** 3 / rho_r ** 2) / (m_r[k] ** 3 / rho_r[k] ** 2))
f_heat = float(np.mean(m_r) / m_r[k])
model_f_fan = float(DRO.fan_electric_kw * C.ID_FAN_MOTOR_EFFICIENCY / DRO.fan_kw)
model_f_heat = float(ORC.orc_mean_kw / ORC.orc_gross_kw)
check("design record is the maximum gas flow", close(m_r[k], D["m_dot_fg"], 1e-12),
      f"{m_r[k]:.2f} kg/s, {len(m_r)} operating records")
check("annual-mean added fan power, as a fraction of the design record", close(f_fan, model_f_fan, 1e-9),
      f"{f_fan:.4f} vs model {model_f_fan:.4f}: Direct-RO {float(DRO.fan_kw):.0f} kW shaft at the design record -> "
      f"{float(DRO.fan_electric_kw):.0f} kW of electricity on average (motor {C.ID_FAN_MOTOR_EFFICIENCY:g})")
check("annual-mean ORC output (heat follows gas flow)", close(f_heat, model_f_heat, 1e-9),
      f"{f_heat:.4f} vs model {model_f_heat:.4f}: {float(ORC.orc_gross_kw):.1f} -> {float(ORC.orc_mean_kw):.1f} kW")
# Part-load fan efficiency. The model holds the fan efficiency constant.
# These fans are single-speed with inlet vanes; their own logged current says
# efficiency falls at part load. Gas power through the existing fans, taking the
# fabric-filter suction as the pressure rise (a lower bound on the fan's head),
# over shaft power from amps at PF 0.85 (which overstates part-load power):
from common.fan_margin import existing_fan_shaft_kw      # noqa: E402  (amps -> kW only)
amps = pd.to_numeric(raw["ID Fan A amps"], errors="coerce").to_numpy()
eta_r = (m_r / rho_r) * (-p_r) / 2 / existing_fan_shaft_kw(amps)       # per fan, kW / kW
f_fan_eta = float(np.mean(m_r ** 3 / rho_r ** 2 * eta_r[k] / eta_r) / (m_r[k] ** 3 / rho_r[k] ** 2))
elec_extra = ((f_fan_eta - f_fan) * float(DRO.fan_kw) / C.ID_FAN_MOTOR_EFFICIENCY
              * float(DRO.preheated_hours) / 1000 * ec.ELECTRICITY_AUD_PER_MWH / float(DRO.water_m3_per_year))
check("part-load fan efficiency", True,
      f"the fans' own current implies their efficiency falls from {eta_r[k]:.2f} at the design record to "
      f"{np.median(eta_r):.2f} at the median record. If the added duty followed that trend the annual fan "
      f"fraction would be {f_fan_eta:.3f}, not {f_fan:.3f}: +A${elec_extra:.3f}/m3 on Direct-RO", flag=True)

# ==========================================================================
say("\nE2  LCOW rebuilt from its parts, every architecture")
# ==========================================================================
worst = 0.0
for basis, df in (("global", g),):
    for _, r in df.iterrows():
        cap = r.total_capex_aud * crf
        is_ro = r.technology.upper() != "MED"
        om = (r.capex_heat_recovery_aud * ec.OANDM_RATE_ORC_HX + r.capex_water_aud * ec.OANDM_RATE_MED_RO)
        # heat side at its annual mean over the records (E2a): ORC with the flow
        # fraction, the fan with the cube-law fraction and the motor, the pump as is
        heat_kw = (r.orc_gross_kw * f_heat - r.fan_kw * f_fan / C.ID_FAN_MOTOR_EFFICIENCY
                   - r.loop_pump_kw)
        mwh = (heat_kw * r.preheated_hours - r.process_kw * r.operating_hours) / 1000
        lcow = (cap + om - mwh * ec.ELECTRICITY_AUD_PER_MWH) / r.water_m3_per_year
        worst = max(worst, abs(lcow / r.lcow_aud_per_m3 - 1))
check(f"LCOW = (CAPEX x CRF + O&M + electricity) / water, {len(g)} rows", worst < 1e-9,
      f"worst relative gap {worst:.1e}; heat-side power at its mean over the records, charged over the "
      f"{hrs_med:,.0f} h with flue gas")

# Consistency, not independence: Direct-RO's loop return comes from an
# annual-cost objective. Scoring every candidate in its scan by the model's own
# full LCOW must pick the same loop return.
import importlib.util as _il                               # noqa: E402
_sp = _il.spec_from_file_location("m6", ROOT / "module6_lcow" / "module6_lcow.py")
m6 = _il.module_from_spec(_sp); _sp.loader.exec_module(m6)
from common.fgche_sizing import size_fgche, gas_side_fan_power, loop_pump_power_kw   # noqa: E402
from common.costing import installed_cost, hx_purchased_cost_aud                     # noqa: E402
_perf = m6.ro_annual_performance(float(DRO.design_heat_kw), C.RO_CEILING_CURRENT_MEMBRANE_C, str(ROOT / "data"))
_tsw, _hot, _ceil = _perf["design_seawater_c"], TRACK - C.LOSS_BUFFER_DELTA_T, C.RO_CEILING_CURRENT_MEMBRANE_C
_gt = m6.gas_side_heat_transfer(D["t_in_c"], D["m_dot_fg"], D["p_gauge"], D["dust"], a_min)
_scan = []
for _lr in np.arange(_tsw + C.ORC_CONDENSER_PINCH_C, _hot - 5.0, 0.5):
    _sz = size_fgche(float(_lr), _hot, q_meanT, D["t_in_c"], c_g, _gt, n_tubes, C.DUCT_HEIGHT_M)
    if _sz is None or _sz["rows_316"] > m6._MED_ROW_CAP:
        continue
    _d1, _d2 = _hot - _ceil, float(_lr) - _tsw
    _area_pre = float(DRO.design_heat_kw) * 1000 / (m6.ORC_U_BASELINE_W_M2K * (_d1 - _d2) / np.log(_d1 / _d2))
    _r = m6.assemble_architecture(
        "scan", "SWRO", float(DRO.design_heat_kw), _perf["water_m3_per_day"], _sz["area_316_m2"],
        gas_side_fan_power(_sz["rows_316"], _gt)["fan_kw"], loop_pump_power_kw(_sz["m_dot_loop_kg_s"]), 0.0,
        _perf["sec_preheated_kwh_m3"], _perf["sec_ambient_kwh_m3"],
        extra_heat_capex_aud=installed_cost(hx_purchased_cost_aud(_area_pre)))
    _scan.append((float(_lr), _r["lcow_aud_per_m3"]))
_best = min(_scan, key=lambda x: x[1])
# (1e-6: this script's mean-temperature duty matches Module 6's to 1e-6, T2)
check("Direct-RO loop return is the LCOW minimum of its own scan", close(_best[0], float(DRO.loop_return_c), 1e-9)
      and close(_best[1], float(DRO.lcow_aud_per_m3), 1e-6),
      f"{len(_scan)} fan-feasible candidates; lowest LCOW A${_best[1]:.4f} at {_best[0]:.2f} C, "
      f"model {float(DRO.loop_return_c):.2f} C")
check("process power = SEC x water", close(float(DRO.process_kw),
      float(DRO.sec_effective_kwh_m3) * float(DRO.water_m3_per_day) / 24, 1e-9),
      f"{float(DRO.process_kw):,.0f} kW at {float(DRO.sec_effective_kwh_m3):.3f} kWh/m3")

# ==========================================================================
say("\nE3  The headline comparison")
# ==========================================================================
same_cap = close(float(PLAIN.water_m3_per_day), float(DRO.water_m3_per_day), 1e-9)
check("conventional SWRO held at Direct-RO's capacity", same_cap and close(float(PLAIN.capex_water_aud),
      float(DRO.capex_water_aud), 1e-9), f"{float(PLAIN.water_m3_per_day):,.0f} m3/day, same desalination CAPEX")
check("ordering conventional SWRO < Direct-RO < ORC+RO",
      PLAIN.lcow_aud_per_m3 < DRO.lcow_aud_per_m3 < ORC.lcow_aud_per_m3,
      f"A${PLAIN.lcow_aud_per_m3:.3f} < {DRO.lcow_aud_per_m3:.3f} < {ORC.lcow_aud_per_m3:.3f}")

# ==========================================================================
say("\nE4  Desalination capital from Bhojwani et al. (2019)")
# ==========================================================================
from common.costing import usd_to_aud, escalate        # noqa: E402  (unit conversions only)
q = float(PLAIN.water_m3_per_day)
spec = float(PLAIN.capex_water_aud) / q
to_aud = usd_to_aud(escalate(1.0, ec.BHOJWANI_BASE_YEAR))
unit_back = spec / (365 * 0.95 * 20) / to_aud / ec.LOCATION_FACTOR_AU   # Section 4.3: 20 yr, 95 %; Australian location factor
unit_t5 = ec.BHOJWANI_SWRO_UNIT_COEFF_USD * q ** ec.BHOJWANI_SCALE_EXPONENT
check("conventional SWRO CAPEX reverses Table 5 with the paper's own annualisation and the location factor", close(unit_back, unit_t5, 1e-9),
      f"A${spec:,.0f} per m3/day -> {unit_back:.4f} USD/m3 of capital, Table 5 fit {unit_t5:.4f} at {q:,.0f} m3/d")
fig4 = usd_to_aud(escalate(0.0046915e6 * q ** 0.87547, ec.BHOJWANI_BASE_YEAR)) / q * ec.LOCATION_FACTOR_AU
check("the paper's Fig. 4 SWRO correlation at the same capacity", abs(fig4 / spec - 1) <= 0.30,
      f"A${fig4:,.0f} per m3/day, {100 * (fig4 / spec - 1):+.0f} % from the Table 5 reversal; Table 5 and Fig. 4 disagree "
      f"with each other, inside the paper's own +/-20-30 %")

say(f"\n{TALLY['PASS']} pass, {TALLY['FLAG']} flag, {TALLY['FAIL']} fail")
(Path(__file__).with_suffix(".log")).write_text("\n".join(LOG) + "\n")
sys.exit(1 if TALLY["FAIL"] else 0)        # run_all stops on a failure
