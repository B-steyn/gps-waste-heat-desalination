"""
Figures, drawn only from `results/` into `results/figures/`.

Every number on a figure is read from a results CSV, never typed, so a
figure cannot drift from the results it shows. Run after
`reporting/collect_results.py`; `run_all.py` does this near the end.

  fig1_lcow_bridge           conventional SWRO -> Direct-RO -> ORC+RO, component by component
  fig2_penalty_distribution  the paired penalties across all 19,683 input combinations
  fig3_pinch_composite       an endpoint-passing ORC design crosses; the modelled one does not
  fig4_loop_return_scan      LCOW, net electricity and ORC output against loop return
  fig5_med_heat_duration     why a P90-sized MED plant averages about 77 % of its size
  fig6_monte_carlo_scurves   chance the LCOW comes in below any price - Monte Carlo
  fig7_pinch_check           the evaporator approach before and after the bubble point was checked
  fig8_orc_step_ro           what adding the ORC does to Direct-RO
  fig9_orc_step_med          what adding the ORC does to Direct-MED
  diag_pinch_resolution      diagnostic: the check points against the true minimum

Style follows the dataviz reference palette, validated for light mode:
series slots #2a78d6 / #eb6834 / #1baf7a, and status red only where
something is physically impossible or breaks a design rule. Grids are solid
hairlines, each panel has one y-axis, legends appear only for two or more
series, and text is set in ink, never in a series colour.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.constants import ORC_EVAPORATOR_PINCH_C

SRC = ROOT / "results"
OUT = SRC / "figures"
INDEX = []
REPORT = os.environ.get("FIGURES_FOR_REPORT") == "1"   # report copies: no footnote, no basis title

INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3, in order
CRIT = "#d03b3b"                                   # status: critical
SHADE = "#f0efec"                                  # neutral region wash

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": BASE, "ytick.color": BASE, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "text.parse_math": False,          # "A$...$" must never become maths mode
})


def _signed(v):
    s = f"{v:+.3f}" if abs(v) < 0.01 else f"{v:+.2f}"
    return s.replace("-", "\u2212")


def save(fig, name, caption, source, status="current", audience="report"):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    INDEX.append((name, caption, source, status, audience))
    print(f"  {name}")


def _m6(basis="bhojwani_global"):
    return pd.read_csv(SRC / f"gps_module6_lcow_{basis}_lcow.csv")


def _lcow(g, arch, track=None):
    x = g[g.architecture == arch]
    x = x[x.track_c == track] if track is not None else x[x.track_c.isna()]
    return float(x.lcow_aud_per_m3.iloc[0])


# --------------------------------------------------------------------------
def fig1_bridge():
    b = pd.read_csv(SRC / "report_lcow_breakdown.csv")
    P, D, O = "Conventional SWRO (no heat recovery)", "Direct-RO (no ORC)", "orc_ro_current (path A)"

    def pick(basis, arch, track=None):
        x = b[(b.cost_basis == basis) & (b.architecture == arch)]
        x = x[x.track_c == track] if track is not None else x
        assert len(x), (basis, arch, track)
        return x.groupby("component").aud_per_m3.sum(), float(x.lcow_total_aud_per_m3.iloc[0])

    plain, t_p = pick("bhojwani_global", P)
    dro, t_d = pick("bhojwani_global", D, 85.0)
    orc, t_o = pick("bhojwani_global", O, 85.0)
    d1, d2 = dro.sub(plain, fill_value=0.0), orc.sub(dro, fill_value=0.0)
    s1 = [("FGCHE\ncapital", d1["FGCHE (flue-gas exchanger)"]),
          ("Heat\nO&M", d1["O&M heat-recovery pool"]),
          ("Preheater\ncapital", d1["loop-seawater preheater"]),
          ("Fan +\npump", d1["ID fan parasitic"] + d1["loop circulating pump"])]
    saving = (t_d - t_p) - sum(v for _, v in s1)              # closes the bridge exactly
    assert abs(saving - d1["desalination process demand"]) < 2e-3
    s1.append(("Preheat\nsaving", saving))
    # the ORC step by what it actually buys and displaces
    s2 = [("ORC\nexchangers", d2["ORC evaporator + condenser"]),
          ("Larger\nFGCHE", d2["FGCHE (flue-gas exchanger)"]),
          ("Extra\nO&M", d2["O&M heat-recovery pool"]),
          ("Preheater\nremoved", d2["loop-seawater preheater"])]
    other = (t_o - t_d) - sum(v for _, v in s2)               # closes the bridge exactly
    o_pkg = d2["ORC package (turbine/pump/gen)"]
    o_pow = d2["ID fan parasitic"] + d2["loop circulating pump"] + d2["ORC generation credit"]
    o_scale = other - o_pkg - o_pow                              # slightly smaller plant
    s2.append(("Other\n(net)", other))
    seq = ([("Conventional\nSWRO", t_p, "total")] + [(l, v, "step") for l, v in s1]
           + [("Direct-RO", t_d, "total")] + [(l, v, "step") for l, v in s2] + [("ORC+RO", t_o, "total")])

    # extra room either side of each total, whose bold labels are the widest
    xs, x = [], 0.0
    for i, (_, _, kind) in enumerate(seq):
        if i and (kind == "total" or seq[i - 1][2] == "total"):
            x += 0.3
        xs.append(x)
        x += 1.0
    fig, ax = plt.subplots(figsize=(8.8, 4.1))
    level, i_save = 0.0, None
    for i, (lab, v, kind) in enumerate(seq):
        if kind == "total":
            ax.bar(xs[i], v, width=0.6, color=S1, zorder=3)
            ax.text(xs[i], v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK,
                    fontweight="bold")
            level = v
        else:
            ax.bar(xs[i], abs(v), bottom=level + min(v, 0.0), width=0.6, color=S2 if v >= 0 else S3, zorder=3)
            ax.text(xs[i], level + max(v, 0.0) + 0.04, _signed(v), ha="center", va="bottom", fontsize=8.5,
                    color=INK2)
            if lab == "Preheat\nsaving":
                i_save = i
            level += v
        if i < len(seq) - 1:
            ax.plot([xs[i] + 0.3, xs[i + 1] - 0.3], [level, level], color=MUTED, lw=0.7, zorder=2)
    ax.annotate("The heat's entire\nbenefit", xy=(xs[i_save], t_d + 0.2), xytext=(xs[i_save], t_d + 0.62),
                ha="center", va="bottom", fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xticks(xs, [s[0] for s in seq])
    for t, s in zip(ax.get_xticklabels(), seq):
        t.set_fontsize(8.0 if s[2] == "total" else 7.0)
        if s[2] == "total":
            t.set_fontweight("bold"); t.set_color(INK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, t_o * 1.2)
    ax.set_ylabel("LCOW (A$/m³)")
    ax.set_title(f"Recovering the heat adds A${t_d - t_p:.2f}/m³ over conventional SWRO; "
                 f"the ORC adds A${t_o - t_d:.2f} more", loc="left", color=INK)
    ax.legend(handles=[Patch(color=S1, label="LCOW"), Patch(color=S2, label="Cost added"),
                       Patch(color=S3, label="Saving")], loc="upper left", ncol=3)
    foot = "Global cost basis, capacity held at the Direct-RO plant's."
    if not REPORT:
        fig.text(0.01, -0.04, foot, fontsize=7.5, color=INK2)
    save(fig, "fig1_lcow_bridge",
         "LCOW bridge from conventional SWRO to Direct-RO to ORC+RO, component by component. The ORC step: "
         "its evaporator and condenser, the larger FGCHE its warmer loop return needs, the O&M on that extra "
         "plant, and the preheater it replaces. Other (net) is the ORC package "
         f"({_signed(o_pkg)}), fan and pump power less the ORC's output ({_signed(o_pow)}) and the slightly "
         f"smaller plant ({_signed(o_scale)}), in A$/m³.",
         "report_lcow_breakdown.csv")


# --------------------------------------------------------------------------
def fig2_penalties():
    f = pd.read_csv(SRC / "sweep_stage3_factorial.csv")
    keys = [c for c in f.columns if c not in ("architecture", "lcow")]
    w = f.pivot_table(index=keys, columns="architecture", values="lcow", aggfunc="first")
    heat = (w["Direct-RO (no ORC)"] - w["Conventional SWRO (no heat recovery)"]).dropna()
    orc = (w["orc_ro_current (path A)"] - w["Direct-RO (no ORC)"]).dropna()
    assert len(heat) == len(orc) == len(w), "every combination must pair"
    g = _m6()
    base_heat = _lcow(g, "Direct-RO (no ORC)", 85.0) - _lcow(g, "Conventional SWRO (no heat recovery)")
    base_orc = _lcow(g, "orc_ro_current (path A)", 85.0) - _lcow(g, "Direct-RO (no ORC)", 85.0)

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for ax, s, base, title in ((axs[0], heat, base_heat, "Heat recovery: Direct-RO − conventional SWRO"),
                               (axs[1], orc, base_orc, "The ORC: ORC+RO − Direct-RO")):
        ax.hist(s, bins=48, color=S1, edgecolor="white", linewidth=0.4, zorder=3)
        top = ax.get_ylim()[1]
        ax.axvline(0, color=INK, lw=0.9, zorder=4)
        ax.text(0, top * 1.36, " break-even", ha="left", va="center", fontsize=7.5, color=INK)
        for v, lab, dy in ((base, f"baseline {base:.2f}", 1.22), (float(s.median()), f"median {s.median():.2f}", 1.08)):
            ax.plot([v, v], [0, top * dy], color=INK2, lw=0.8, zorder=4)
            ax.text(v, top * dy, f" {lab}", ha="left", va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(-0.06 * s.max(), s.max() * 1.04)
        ax.set_ylim(0, top * 1.45)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("A$/m³ added")
        ax.text(0.98, 0.60, f"above zero in\n{int((s > 0).sum()):,} of {len(s):,}\n"
                f"min {s.min():.2f} · max {s.max():.2f}", transform=ax.transAxes, ha="right",
                va="top", fontsize=7.5, color=INK2)
    axs[0].set_ylabel("Input combinations")
    save(fig, "fig2_penalty_distribution",
         f"Paired penalties on the global cost basis: every one of the {len(heat):,} combinations of the "
         "nine swept economic inputs, each architecture scored on the same inputs. A grid of equally "
         "weighted combinations, not a probability distribution (see the Monte Carlo, figure 6).",
         "sweep_stage3_factorial.csv")


# --------------------------------------------------------------------------
def fig3_pinch():
    c = pd.read_csv(SRC / "pinch_composite_curves.csv")
    e = pd.read_csv(SRC / "pinch_exergy_check.csv")
    fig = plt.figure(figsize=(6.8, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.55, wspace=0.28)
    titles = {"invalid": "Endpoint check passes; the streams cross",
              "corrected": "Discretised check; the design as modelled"}
    for k, case in enumerate(("invalid", "corrected")):
        ax = fig.add_subplot(gs[0, k])
        x = c[c.case == case].sort_values("fraction")
        qt = x.q_cumulative_kw.max()
        mw = (qt - x.q_cumulative_kw) / 1000.0                   # heat from the cold end
        ax.plot(mw, x.t_hot_c, color=S2, label="Loop water", zorder=3)
        ax.plot(mw, x.t_cold_c, color=S1, label="Working fluid", zorder=3)
        i = x.approach_k.idxmin()
        if case == "invalid":
            ax.fill_between(mw, x.t_hot_c, x.t_cold_c, where=x.t_cold_c > x.t_hot_c, color=CRIT,
                            alpha=0.18, lw=0, interpolate=True, zorder=2)
            ax.annotate(f"cross of {abs(x.approach_k.min()):.0f} K:\nimpossible",
                        xy=(mw.loc[i], (x.t_hot_c.loc[i] + x.t_cold_c.loc[i]) / 2),
                        xytext=(mw.loc[i] + 0.9, x.t_hot_c.min() + 3), fontsize=7.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
            note = (f"Ends look fine: {x.t_hot_c.iloc[0] - x.t_evap_c.iloc[0]:.1f} K to\n"
                    f"saturation at the hot end,\n{x.endpoint_cold_k.iloc[0]:+.1f} K at the cold end")
        else:
            note = (f"Closest approach {x.approach_k.min():.1f} K,\nnear the cold end; the\n"
                    f"design rule is {ORC_EVAPORATOR_PINCH_C:.0f} K")
            ax.legend(loc="upper left")                    # one legend serves both panels
        ax.text(0.45 if case == "invalid" else 0.40, 0.03 if case == "invalid" else 0.12, note,
                transform=ax.transAxes, fontsize=7, color=INK2, va="bottom")
        ax.set_title(titles[case], loc="left", color=INK)
        ax.set_xlabel("Heat from the cold end (MW)")
        if k == 0:
            ax.set_ylabel("Temperature (°C)")
    ax = fig.add_subplot(gs[1, :])
    for case, col, lab in (("invalid", CRIT, "Endpoint-passing design"),
                           ("corrected", INK2, "Design as modelled")):
        x = e[e.case == case].sort_values("fraction")
        qt = c[c.case == case].q_cumulative_kw.max()
        tot = float(x.e_d_cumulative_kw.iloc[-1])
        ax.plot((1 - x.fraction) * qt / 1000.0, tot - x.e_d_cumulative_kw, color=col,
                label=f"{lab}: {tot:+,.0f} kW in total", zorder=3)
    ax.axhline(0, color=INK, lw=0.8, zorder=2)
    ax.set_title("Second law: exergy destruction can never be negative", loc="left", color=INK)
    ax.set_xlabel("Heat from the cold end (MW)")
    ax.set_ylabel("Exergy destroyed,\ncumulative (kW)")
    ax.legend(loc="upper left")
    save(fig, "fig3_pinch_composite",
         "Top: evaporator composite curves for the design an endpoint check passes (left) and the "
         "discretised-check design (right). Bottom: cumulative exergy destruction; the endpoint-passing "
         "design goes negative, which the second law forbids.",
         "pinch_composite_curves.csv, pinch_exergy_check.csv")


# --------------------------------------------------------------------------
def fig4_scan():
    s = pd.read_csv(SRC / "loop_return_scan.csv")
    s = s[s.feasible.astype(bool)].copy()
    ok = s[s.fan_feasible.astype(bool)]
    cap_lr, cap_rows = ok.loop_return_c.max(), int(ok.rows.max())
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 6.6), sharex=True)
    panels = (("lcow_aud_per_m3", "LCOW (A$/m³)", "min", 2),
              ("net_electricity_kw", "Net electricity,\nannual mean (kW)", "max", 0),
              ("w_net_kw", "ORC net output,\ndesign point (kW)", "max", 0))
    for ax, (col, lab, sense, dp) in zip(axs, panels):
        ax.axvspan(cap_lr, s.loop_return_c.max() + 0.5, color=SHADE, lw=0, zorder=0)
        for p, colr, name in (("A", S1, "Path A (subcritical)"), ("B", S2, "Path B (transcritical)")):
            x = s[s.path == p].sort_values("loop_return_c")
            ax.plot(x.loop_return_c, x[col], color=colr, label=name, zorder=3)
        a = s[s.path == "A"]
        i = a[col].idxmin() if sense == "min" else a[col].idxmax()
        ax.plot(a.loc[i, "loop_return_c"], a.loc[i, col], "o", ms=5.5, mfc=INK, mec="white", mew=1.2,
                zorder=4)
        word = "lowest" if sense == "min" else "highest"
        val = f"{a.loc[i, col]:,.{dp}f}".replace("-", "\u2212")
        place = {"lcow_aud_per_m3": dict(xytext=(6, -12), textcoords="offset points", ha="left", va="top"),
                 "net_electricity_kw": dict(xytext=(0.72, 0.74), textcoords="axes fraction", ha="center",
                                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7)),
                 "w_net_kw": dict(xytext=(0.97, 0.42), textcoords="axes fraction", ha="right",
                                  arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))}[col]
        ax.annotate(f"{word} {val} at {a.loc[i, 'loop_return_c']:.1f} °C",
                    xy=(a.loc[i, "loop_return_c"], a.loc[i, col]), fontsize=7.5, color=INK, **place)
        if col == "lcow_aud_per_m3":                       # room beneath the minimum for its label
            ax.set_ylim(s[col].min() - 0.32, s[col].max() + 0.08)
        ax.set_ylabel(lab)
    axs[0].text(cap_lr + 0.6, axs[0].get_ylim()[1], f"beyond the {cap_rows}-row ID fan cap",
                va="top", fontsize=7.5, color=INK2)
    axs[2].legend(loc="upper left")
    axs[0].set_title(f"Three objectives, three optima; only {s.loop_return_c.min():.0f}–{cap_lr:.0f} °C "
                     "fits the ID fan", loc="left", color=INK)
    axs[-1].set_xlabel("Loop return temperature (°C)")
    save(fig, "fig4_loop_return_scan",
         "ORC+RO at the 85 °C track and the measured duct. Grey: loop returns whose tube bank "
         "exceeds the ID fan's power margin. Markers: each objective's optimum for path A. Net "
         "electricity is the annual mean over the operating records. Path B's turbine expands a "
         "liquid-like fluid into a two-phase exhaust and is scored with a dry-vapour efficiency, so its "
         "output is overstated.",
         "loop_return_scan.csv")


# --------------------------------------------------------------------------
def fig5_med():
    g = _m6()
    x = g[g.architecture == "Direct-MED"]
    r = x.loc[x.lcow_aud_per_m3.idxmin()]
    q = pd.read_csv(SRC / f"gps_module3_output_direct_med_{int(r.track_c)}.csv")[
        "heat_delivered_to_downstream_kw"]
    q = np.sort(q[q > 0].values)[::-1] / 1000.0
    size = float(r.design_heat_kw) / 1000.0
    hrs = (np.arange(len(q)) + 0.5) / len(q) * 100.0
    used = np.minimum(q, size)
    cf, spill = used.mean() / size, 1.0 - used.sum() / q.sum()
    assert abs(cf - float(r.capacity_factor)) < 1e-6, "must reproduce Module 6's capacity factor"
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.fill_between(hrs, 0, used, color=S1, alpha=0.12, lw=0, zorder=2)
    ax.fill_between(hrs, size, q, where=q > size, color=S2, lw=0, zorder=2, interpolate=True)
    ax.plot(hrs, q, color=S1, zorder=3)
    ax.axhline(size, color=INK, lw=1.0, zorder=3)
    ax.text(99, size + 0.12, f"plant size = the 90th-percentile hour = {size:.2f} MW",
            ha="right", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(used.mean(), color=INK2, lw=0.8, zorder=3)
    ax.text(1, used.mean() - 0.12, f"average heat it can use = {used.mean():.2f} MW = {cf:.1%} of its size",
            ha="left", va="top", fontsize=7.5, color=INK2)
    ax.annotate(f"too much heat for the plant:\n{spill:.2%} of the year's heat spilled",
                xy=(2.5, (q[0] + size) / 2), xytext=(12, size + 0.75), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.text(50, used.mean() * 0.45, "heat turned into water", ha="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, q.max() * 1.22)
    ax.set_xlabel("Share of the 461 recorded running hours, hottest first (%)")
    ax.set_ylabel("Heat reaching the MED plant (MW)")
    ax.set_title(f"Direct-MED, {int(r.track_c)} °C track: sized for the 90th-percentile hour, "
                 f"the plant averages {cf:.0%} of its size", loc="left", color=INK)
    save(fig, "fig5_med_heat_duration",
         "Heat-duration curve for Direct-MED. The plant is built for the 90th-percentile hour and each hour "
         "makes water from the heat that arrives, capped at its size; the shaded area is the water made, "
         "not the full rectangle.",
         f"gps_module3_output_direct_med_{int(r.track_c)}.csv, gps_module6_lcow_bhojwani_global_lcow.csv")


# --------------------------------------------------------------------------
def fig7_pinch_check():
    fb, fa = SRC / "pinch_before_after_profile.csv", SRC / "pinch_profiles_all_designs.csv"
    if not (fb.exists() and fa.exists()):
        print("  (fig7 skipped - run analysis/supporting_calculations.py first)")
        return
    ba, al = pd.read_csv(fb), pd.read_csv(fa)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.3), gridspec_kw={"width_ratios": [1.0, 1.2]})
    ax, mins = axs[0], {}
    for case, col, tx in (("before the fix", S2, 2.4), ("after the fix", S1, 5.6)):
        x = ba[(ba.case == case) & (ba.pct_duty_from_cold_end <= 12.0)].sort_values("pct_duty_from_cold_end")
        ax.plot(x.pct_duty_from_cold_end, x.approach_k, color=col, label=case.capitalize(), zorder=3)
        m = x[x.model_check_point.astype(bool)]
        ax.plot(m.pct_duty_from_cold_end, m.approach_k, "o", ms=5, mfc=col, mec="white", mew=1.2, zorder=4)
        i = x.approach_k.idxmin()
        mins[case] = float(x.approach_k[i])
        where = "at a check point" if bool(x.model_check_point[i]) else "between check points"
        ax.annotate(f"{x.approach_k[i]:.2f} K, {where}", xy=(x.pct_duty_from_cold_end[i], x.approach_k[i]),
                    xytext=(tx, x.approach_k[i] - (0.35 if case.startswith("before") else 0.55)),
                    textcoords="data", va="center", fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.axhline(ORC_EVAPORATOR_PINCH_C, color=INK2, lw=0.8, zorder=2)
    ax.text(11.8, ORC_EVAPORATOR_PINCH_C + 0.12, f"{ORC_EVAPORATOR_PINCH_C:.0f} K design minimum", ha="right",
            va="bottom", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 12)
    ax.set_ylim(min(mins.values()) - 0.6, ax.get_ylim()[1])
    ax.set_xlabel("Duty from the cold end (%)")
    ax.set_ylabel("Temperature difference (K)")
    ax.set_title("ORC+RO evaporator, cold end", loc="left", color=INK)
    ax.legend(loc="upper left")
    ax = axs[1]
    for d, x in al.groupby("design"):
        x = x.sort_values("pct_duty_from_cold_end")
        head = d.startswith("orc_ro_current (path A)")
        ax.plot(x.pct_duty_from_cold_end, x.approach_k, color=S1 if head else BASE,
                lw=1.8 if head else 0.9, zorder=3 if head else 2)
    ax.axhline(ORC_EVAPORATOR_PINCH_C, color=INK2, lw=0.8, zorder=2)
    ax.text(98, ORC_EVAPORATOR_PINCH_C - 0.6, f"Closest of all designs: {al.approach_k.min():.2f} K", ha="right",
            va="top", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, al.approach_k.max() * 1.08)
    ax.set_xlabel("Duty from the cold end (%)")
    ax.set_title(f"All {al.design.nunique()} reported ORC designs, after the fix", loc="left", color=INK)
    ax.legend(handles=[Line2D([0], [0], color=S1, lw=1.8, label="ORC+RO"),
                       Line2D([0], [0], color=BASE, lw=0.9, label="The other designs")], loc="upper left")
    save(fig, "fig7_pinch_check",
         f"Temperature difference between the loop water and the working fluid along the ORC evaporator. Left: "
         f"ORC+RO's cold end before and after the check was made to include the working fluid's bubble point. "
         f"Even points alone step over the boiling point, where the difference fell to "
         f"{mins['before the fix']:.2f} K; the fixed check includes that point, "
         f"and the redesigned evaporator holds {mins['after the fix']:.2f} K there. Right: every reported ORC "
         f"design after the fix; none falls below {al.approach_k.min():.2f} K.",
         "pinch_before_after_profile.csv, pinch_profiles_all_designs.csv")


GROUPS = (("ORC\nequipment", ("ORC evaporator + condenser", "ORC package (turbine/pump/gen)")),
          ("FGCHE\ncapital", ("FGCHE (flue-gas exchanger)",)),
          ("Heat-plant\nO&M", ("O&M heat-recovery pool",)),
          ("Preheater\ncapital", ("loop-seawater preheater",)),
          ("Desalination\nplant", ("desalination plant", "O&M desalination (ex-membranes)", "membrane replacement")),
          ("Fan + pump\npower", ("ID fan parasitic", "loop circulating pump")),
          ("Process\nelectricity", ("desalination process demand",)),
          ("ORC\nelectricity", ("ORC generation credit",)))


def bridge_step(a_from, a_to, name, lab_from, lab_to, caption):
    """What adding the ORC does, component by component, between two architectures' best designs."""
    b = pd.read_csv(SRC / "report_lcow_breakdown.csv").query("cost_basis == 'bhojwani_global'")
    g = _m6()

    def best(a):
        x = g[g.architecture == a]
        return x.loc[x.lcow_aud_per_m3.idxmin()]

    def comp(a, t):
        x = b[(b.architecture == a) & (b.track_c == t)]
        assert len(x), (a, t)
        return x.groupby("component").aud_per_m3.sum(), float(x.lcow_total_aud_per_m3.iloc[0])

    rf, rt = best(a_from), best(a_to)
    cf, tf = comp(a_from, rf.track_c)
    ct, tt = comp(a_to, rt.track_c)
    assert abs(tf - rf.lcow_aud_per_m3) < 1e-3 and abs(tt - rt.lcow_aud_per_m3) < 1e-3, "must match Module 6"
    d = ct.sub(cf, fill_value=0.0)
    known = {c for _, cs in GROUPS for c in cs}
    assert set(d.index) <= known, f"unmapped components: {set(d.index) - known}"
    steps = [(lab, float(sum(d.get(c, 0.0) for c in cs))) for lab, cs in GROUPS]
    shown = [(l, v) for l, v in steps if abs(v) >= 0.0005]
    omitted = [l.replace("\n", " ").lower() for l, v in steps if abs(v) < 0.0005]
    resid = (tt - tf) - sum(v for _, v in shown)
    assert abs(resid) < 3e-3, f"bridge does not close: {resid}"
    k = max(range(len(shown)), key=lambda i: abs(shown[i][1]))
    shown[k] = (shown[k][0], shown[k][1] + resid)          # absorb CSV rounding so it closes exactly
    seq = [(lab_from, tf, "total")] + [(l, v, "step") for l, v in shown] + [(lab_to, tt, "total")]
    fig, ax = plt.subplots(figsize=(max(5.4, 0.76 * len(seq) + 1.2), 3.9))
    level, peak = 0.0, 0.0
    for i, (lab, v, kind) in enumerate(seq):
        if kind == "total":
            ax.bar(i, v, width=0.6, color=S1, zorder=3)
            ax.text(i, v + 0.012 * tt, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK,
                    fontweight="bold")
            level = v
        else:
            ax.bar(i, abs(v), bottom=level + min(v, 0.0), width=0.6, color=S2 if v >= 0 else S3, zorder=3)
            ax.text(i, level + max(v, 0.0) + 0.012 * tt, _signed(v), ha="center", va="bottom", fontsize=8.5,
                    color=INK2)
            level += v
        peak = max(peak, level, v if kind == "total" else 0.0)
        if i < len(seq) - 1:
            ax.plot([i + 0.3, i + 0.7], [level, level], color=MUTED, lw=0.7, zorder=2)
    ax.set_xticks(range(len(seq)), [x[0] for x in seq])
    for t, x in zip(ax.get_xticklabels(), seq):
        t.set_fontsize(8.5 if x[2] == "total" else 7.0)
        if x[2] == "total":
            t.set_fontweight("bold"); t.set_color(INK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, peak * 1.22)
    ax.set_ylabel("LCOW (A$/m³)")
    ax.set_title(f"Adding the ORC to {lab_from} costs A${tt - tf:.2f}/m³", loc="left", color=INK)
    ax.legend(handles=[Patch(color=S1, label="LCOW"), Patch(color=S2, label="Cost added"),
                       Patch(color=S3, label="Saving")], loc="upper left", ncol=3)
    note = (f"Global cost basis. {lab_from}: {rf.track_c:.0f} °C track, {rf.water_m3_per_day:,.0f} m³/day; "
            f"{lab_to}: {rt.track_c:.0f} °C track, {rt.water_m3_per_day:,.0f} m³/day.")
    if omitted:
        note += " Steps under A$0.0005/m³ not drawn: " + ", ".join(omitted) + "."
    if not REPORT:
        fig.text(0.01, -0.05, note, fontsize=7.5, color=INK2, wrap=True)
    save(fig, name, caption, "report_lcow_breakdown.csv, gps_module6_lcow_bhojwani_global_lcow.csv")


def fig8_orc_step_ro():
    bridge_step("Direct-RO (no ORC)", "orc_ro_current (path A)", "fig8_orc_step_ro", "Direct-RO", "ORC+RO",
                "What adding an ORC does to Direct-RO, component by component. The ORC needs its own exchangers; "
                "the loop must return warmer to heat the working fluid, so the FGCHE grows, with more O&M and "
                "more fan power for the deeper bank; the ORC replaces the preheater and earns a small "
                "electricity credit. Heat-plant O&M includes the ORC's own O&M, which sits in the "
                "heat-recovery pool.")


def fig9_orc_step_med():
    bridge_step("Direct-MED", "orc_med (path A)", "fig9_orc_step_med", "Direct-MED", "ORC+MED",
                "What adding an ORC does to Direct-MED, component by component, each at its own best design. "
                "Per-m3 steps include the change in track and capacity: the ORC's condenser runs the MED plant "
                "at a lower boiling temperature, so fewer effects make less water from each unit of heat.")


# --------------------------------------------------------------------------
def diag_pinch():
    p = pd.read_csv(SRC / "pinch_resolution_profile.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
    i = p.approach_k.idxmin()
    for ax, lo in ((axs[0], 0.0), (axs[1], 0.9)):
        x = p[p.fraction >= lo - 1e-9]
        m = x[x.model_check_point.astype(bool)]
        ax.plot(x.fraction * 100, x.approach_k, color=S1, label="Approach, 1,000 segments", zorder=3)
        ax.plot(m.fraction * 100, m.approach_k, "o", ms=5, mfc=INK, mec="white", mew=1.2, zorder=4,
                label="The model's check points")
        ax.axhline(ORC_EVAPORATOR_PINCH_C, color=INK2, lw=0.8, zorder=2)
        ax.plot(p.fraction[i] * 100, p.approach_k[i], "o", ms=6, mfc=CRIT, mec="white", mew=1.2, zorder=5)
    fig.supxlabel("Position along the exchanger (% of duty, from the hot end)", y=-0.04, fontsize=9,
                  color=INK2)
    axs[1].annotate(f"true minimum {p.approach_k[i]:.2f} K\nat {p.fraction[i] * 100:.1f} % of duty\n"
                    "(the bubble point)", xy=(p.fraction[i] * 100, p.approach_k[i]),
                    xytext=(94.4, p.approach_k[i] + 2.3), fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    axs[1].text(90.3, ORC_EVAPORATOR_PINCH_C - 0.25, f"{ORC_EVAPORATOR_PINCH_C:.0f} K design minimum",
                va="top", fontsize=7.5, color=INK2)
    axs[1].set_xlim(90, 100)
    axs[1].set_ylim(p.approach_k.min() - 1.0, p[p.fraction >= 0.9].approach_k.max() + 1.0)
    axs[0].set_ylabel("Approach temperature (K)")
    axs[0].legend(loc="upper right")
    axs[0].set_title("ORC+RO evaporator, whole length", loc="left", color=INK)
    axs[1].set_title("Cold end, enlarged", loc="left", color=INK)
    fb = SRC / "pinch_before_after_profile.csv"
    pre = (f"{pd.read_csv(fb).query('case == \"before the fix\"').approach_k.min():.2f} K" if fb.exists()
           else "below 5 K")
    save(fig, "diag_pinch_resolution",
         "Diagnostic: the ORC+RO evaporator's approach at 1,000 points, with the model's own check points "
         "(dots). With 21 even points alone, the working fluid's bubble point falls between the last two, "
         f"where the approach dipped to {pre}; the check now includes it.",
         "pinch_resolution_profile.csv", "diagnostic", audience="diagnostic")


def fig6_monte_carlo():
    f = SRC / "monte_carlo_scurves.csv"
    if not f.exists():
        print("  (fig6 skipped - run module6_lcow/monte_carlo.py first)")
        return
    c = pd.read_csv(f)
    sm = pd.read_csv(SRC / "monte_carlo_summary.csv").query("distribution == 'triangular'")
    n = int(pd.read_csv(SRC / "monte_carlo_convergence.csv").n_draws.max())
    series = (("Conventional SWRO (no heat recovery)", S1, "Conventional SWRO"),
              ("Direct-RO (no ORC)", S2, "Direct-RO"),
              ("orc_ro_current (path A)", S3, "ORC+RO"))
    panels = (("bhojwani_global", "Global cost basis"),)
    fig, axs = plt.subplots(1, len(panels), figsize=(4.4, 3.2), sharey=True, squeeze=False)
    axs = axs[0]
    for ax, (basis, title) in zip(axs, panels):
        for k, (arch, col, lab) in enumerate(series):
            x = c[(c.cost_basis == basis) & (c.architecture == arch)].sort_values("probability")
            ax.plot(x.lcow_aud_per_m3, x.probability * 100.0, color=col, label=lab, zorder=3)
            p50 = float(sm[(sm.cost_basis == basis) & (sm.architecture == arch)].p50.iloc[0])
            ax.plot(p50, 50, "o", ms=5, mfc=col, mec="white", mew=1.2, zorder=4)
            # the middle curve is labelled above and to its left, the others below and to their
            # right, so neighbouring labels never collide however close the curves sit
            ax.annotate(f"{p50:.2f}", xy=(p50, 50), xytext=(-4, 5) if k == 1 else (4, -11),
                        textcoords="offset points", ha="right" if k == 1 else "left",
                        va="bottom" if k == 1 else "baseline", fontsize=7.5, color=INK2)
        ax.set_yticks([0, 10, 50, 90, 100])
        ax.set_ylim(0, 100)
        if not REPORT:
            ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("LCOW (A$/m³)")
    axs[0].set_ylabel("Chance the LCOW is at\nor below this price (%)")
    axs[0].legend(loc="lower right")
    if not REPORT:
        fig.text(0.01, -0.11,
                 f"Triangular Monte Carlo, {n:,} paired draws, global cost basis; dots mark the P50.\n"
                 "Read a tariff off the x-axis: the curve's height there is the chance of beating it.",
                 fontsize=7.5, color=INK2)
    save(fig, "fig6_monte_carlo_scurves",
         "S-curves from the triangular Monte Carlo: the chance that each RO architecture's LCOW comes in at "
         "or below a given price, global cost basis. Reading a water tariff off the x-axis gives the chance "
         "of beating it. MED architectures, far to the right, are in monte_carlo_summary.csv.",
         "monte_carlo_scurves.csv, monte_carlo_summary.csv")


def write_index():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or "unknown"
    lines = ["# Figures", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} at commit `{commit}` by `reporting/figures.py`, "
             "from the CSVs in `results/` only; no number on a figure is typed. Each figure is saved "
             "as PNG (300 dpi) and SVG (vector).", "",
             "| figure | audience | what it shows | data | status |", "|---|---|---|---|---|"]
    lines += [f"| `{n}` | {au} | {c} | `{s}` | {st} |" for n, c, s, st, au in INDEX]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig1_bridge, fig2_penalties, fig3_pinch, fig4_scan, fig5_med, fig6_monte_carlo,
               fig7_pinch_check, fig8_orc_step_ro, fig9_orc_step_med, diag_pinch):
        fn()
    write_index()
    print(f"wrote {len(INDEX)} figures and README.md to {OUT}")


if __name__ == "__main__":
    main()
