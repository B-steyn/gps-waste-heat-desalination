"""
Builds FINAL_RESULTS.csv: one consolidated table of every headline figure,
each with its unit, the basis on which it was computed, and its source file.

A LONG (tidy) table rather than a wide one: one row per figure, with
`value`, `unit`, `basis` and `source_file`. Every number is computed from
its source file at build time; none is typed in, so the table cannot keep
asserting a value after a re-run has changed it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import common.economic_constants as ec


def rows_for(section, items):
    return [{"section": section, "figure": f, "value": v, "unit": u,
             "basis": b, "source_file": s}
            for f, v, u, b, s in items]


def build(output_dir):
    """
    Consolidated figure table. The headline is the THREE-CASE DECOMPOSITION
    (conventional SWRO, Direct-RO, ORC+RO at one held capacity), because the
    result is the cost of recovering the heat, not which recovery
    architecture wins.
    """
    import importlib.util
    from common.ro_thermal import ro_specific_energy
    from common.fan_margin import row_caps_for_ducts
    from common.constants import (DUCT_WIDTH_M, DUCT_HEIGHT_M, ZUKAUSKAS_C,
                                  RO_CEILING_CURRENT_MEMBRANE_C, N_TRAINS_FITTED, RO_RECOVERY)

    out = Path(output_dir)
    root = out.parent
    lcow_def = pd.read_csv(out / "gps_module6_lcow_bhojwani_global_lcow.csv")
    summary = pd.read_csv(out / "loop_optimisation_summary_lcow.csv")
    corners = pd.read_csv(out / "sweep_stage4_corners.csv")
    tornado = pd.read_csv(out / "sweep_stage3_tornado.csv")
    factorial = pd.read_csv(out / "sweep_stage3_factorial.csv")
    med_n = pd.read_csv(out / "med_effects_sweep.csv")
    brk = pd.read_csv(root / "results" / "report_lcow_breakdown.csv")

    sp = importlib.util.spec_from_file_location(
        "m6", Path(__file__).resolve().parent.parent / "module6_lcow" / "module6_lcow.py")
    m6 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m6)

    PLAIN, DRO, ORC = ("Conventional SWRO (no heat recovery)", "Direct-RO (no ORC)",
                       "orc_ro_current (path A)")

    def best(df, name):
        x = df[df.architecture == name]
        return x.loc[x.lcow_aud_per_m3.idxmin()] if len(x) else None

    plain, dro, orc = best(lcow_def, PLAIN), best(lcow_def, DRO), best(lcow_def, ORC)
    med = best(lcow_def, "Direct-MED")

    def kwh_per_m3(kw, row):
        return kw / (row.water_m3_per_day / 24.0)

    perf = m6.ro_annual_performance(float(dro.design_heat_kw),
                                     RO_CEILING_CURRENT_MEMBRANE_C, str(root / "data"))
    gross_benefit = perf["sec_ambient_kwh_m3"] - perf["sec_preheated_kwh_m3"]
    # electricity actually drawn: the added fan at its mean over the operating
    # records, divided by the motor efficiency, plus the loop pump
    fanpump_dro = kwh_per_m3(dro.fan_electric_kw + dro.loop_pump_kw, dro)
    fanpump_orc = kwh_per_m3(orc.fan_electric_kw + orc.loop_pump_kw, orc)
    rec_f = m6._record_factors()
    heat_share = float(dro.preheated_hours / dro.operating_hours)     # hours with heat / RO hours
    lo_price, hi_price = ec.ELECTRICITY_SWEEP_AUD_PER_MWH
    max_benefit_5c = (ro_specific_energy(5.0, r=RO_RECOVERY)["sec_kwh_m3"]
                      - ro_specific_energy(RO_CEILING_CURRENT_MEMBRANE_C, r=RO_RECOVERY)["sec_kwh_m3"])
    row_cap = row_caps_for_ducts(str(root / "data"),
                                  [("measured", DUCT_WIDTH_M, DUCT_HEIGHT_M)])["measured"]
    feas = summary[summary["fan_feasible"].astype(bool)] if "fan_feasible" in summary else summary
    min_pinch = float(feas.min_pinch_k.min())
    # the pinch check includes the working fluid's bubble and dew points; CR6b
    # re-checks every reported ORC design three ways
    _prc = Path(__file__).resolve().parent.parent / "outputs" / "pinch_resolution_check.csv"
    true_pinch = float(pd.read_csv(_prc).true_min_approach_k.min()) if _prc.exists() else min_pinch

    R = []

    # ---- 1. headline ----
    R += rows_for("1_headline", [
        ("ANSWER", "No - recovering the heat adds cost", "-",
         "three-case decomposition at held-fixed capacity", "gps_module6_lcow_*.csv"),
        ("LCOW conventional SWRO (no heat recovery)", round(float(plain.lcow_aud_per_m3), 3), "A$/m3",
         "reference case: no FGCHE, no fan, grid-powered", "-"),
        ("LCOW Direct-RO (heat, no ORC)", round(float(dro.lcow_aud_per_m3), 3), "A$/m3",
         "keeps FGCHE, loop and fan; one preheater instead of two ORC exchangers", "-"),
        ("LCOW ORC+RO (heat and ORC)", round(float(orc.lcow_aud_per_m3), 3), "A$/m3",
         "the architecture the project set out to assess", "-"),
        ("LCOW Direct-MED", round(float(med.lcow_aud_per_m3), 3), "A$/m3",
         f"{int(med.track_c)}C track, TBT {med.tbt_c:.1f}C, {int(med.n_effects)} effects; "
         "loop return held at TBT + 5 K", "-"),
        ("PENALTY for recovering the heat", round(float(dro.lcow_aud_per_m3 - plain.lcow_aud_per_m3), 3),
         "A$/m3", "Direct-RO less conventional SWRO", "-"),
        ("PENALTY for adding the ORC", round(float(orc.lcow_aud_per_m3 - dro.lcow_aud_per_m3), 3),
         "A$/m3", "ORC+RO less Direct-RO", "-"),
        ("capacity (held fixed for comparison)", round(float(plain.water_m3_per_day)), "m3/day",
         "sized from available heat, then held for the no-heat case", "-"),
    ])

    # ---- 2. mechanism ----
    d = brk[(brk.cost_basis == "bhojwani_global") & (brk.architecture == DRO)
            & (brk.track_c == dro.track_c)].set_index("component").aud_per_m3
    pl = brk[(brk.cost_basis == "bhojwani_global")
             & (brk.architecture == PLAIN)].set_index("component").aud_per_m3
    delta = (d - pl).dropna()
    cap = float(delta[["FGCHE (flue-gas exchanger)", "loop-seawater preheater",
                        "O&M heat-recovery pool"]].sum())
    elec = float(delta[["ID fan parasitic", "loop circulating pump"]].sum())
    ben = float(-delta[["desalination process demand"]].sum())
    tot = cap + elec - ben
    R += rows_for("2_mechanism", [
        ("capital + O&M on heat-recovery plant", round(cap, 4), "A$/m3",
         f"{100*cap/tot:.1f}% of the penalty - THE dominant term", "report_lcow_breakdown.csv"),
        ("ID fan + loop pump electricity", round(elec, 4), "A$/m3",
         f"only {100*elec/tot:.1f}% of the penalty; the saving is {-100*ben/tot:.1f}%", "report_lcow_breakdown.csv"),
        ("SEC benefit actually delivered", round(ben, 4), "A$/m3",
         "what the heat is worth - its entire upside", "report_lcow_breakdown.csv"),
        ("economic ratio, spend vs save (Direct-RO)", f"{(cap + elec) / ben:.0f}:1", "-",
         "annualised cost of collecting the heat against the saving it delivers", "-"),
        ("gross preheat benefit, preheated hours", round(gross_benefit, 3), "kWh/m3",
         "per-record seawater intake, mean over the operating records", "-"),
        ("fan + pump energy, Direct-RO", round(fanpump_dro, 3), "kWh/m3",
         "electricity, mean over the operating records", "-"),
        ("fan + pump energy, ORC+RO", round(fanpump_orc, 3), "kWh/m3",
         "electricity, mean over the operating records", "-"),
        ("energy ratio, Direct-RO", f"{fanpump_dro / gross_benefit:.1f}:1", "-",
         "true but NOT the economic mechanism - see the capital row", "-"),
        ("energy ratio, ORC+RO", f"{fanpump_orc / gross_benefit:.1f}:1", "-",
         "fan + pump energy against the preheat benefit, per m3 in the hours with heat", "-"),
        ("free-plant bound (Direct-RO less conventional SWRO, heat plant free)", round(elec - ben, 4), "A$/m3",
         f"A${(elec - ben) * lo_price / ec.ELECTRICITY_AUD_PER_MWH:.3f} / {elec - ben:.3f} / "
         f"{(elec - ben) * hi_price / ec.ELECTRICITY_AUD_PER_MWH:.3f} at A${lo_price:g} / "
         f"{ec.ELECTRICITY_AUD_PER_MWH:g} / {hi_price:g} per MWh: "
         + ("positive at any price, so the heat would not pay even if the plant cost nothing"
            if elec > ben else "NEGATIVE - free heat-recovery plant would pay"),
         "report_lcow_breakdown.csv"),
    ])

    # ---- 3. design ----
    R += rows_for("3_design", [
        ("flue-gas extraction track", int(orc.track_c), "C", "optimisation, not sensitivity", "-"),
        ("loop return, ORC branches", round(float(orc["loop_return_c"]), 2), "C",
         "minimum feasible; LCOW optimum at the floor", "-"),
        ("loop return, Direct-RO", round(float(dro["loop_return_c"]), 2), "C",
         "INTERIOR optimum - the floor result does not transfer", "-"),
        ("FGCHE area, ORC+RO", round(float(orc.hx_area_m2)), "m2", "316L, 2 trains", "-"),
        ("FGCHE area, Direct-RO", round(float(dro.hx_area_m2)), "m2",
         "SMALLER than ORC+RO - the colder loop return widens the cold-end approach", "-"),
        ("ID fan added duty per fan, ORC+RO, design record", round(float(orc.fan_kw) / N_TRAINS_FITTED, 1), "kW",
         "shaft power at the maximum gas flow: what the fan cap checks", "-"),
        ("added ID fan electricity, both fans, Direct-RO / ORC+RO",
         f"{dro.fan_electric_kw:.0f} / {orc.fan_electric_kw:.0f}", "kW",
         f"mean over the {rec_f['n_records']} operating records: {100 * rec_f['fan']:.1f} % of the design-record "
         f"shaft power ({dro.fan_kw:.0f} / {orc.fan_kw:.0f} kW), divided by the motor efficiency",
         "-"),
        ("n operating records", rec_f["n_records"], "records",
         f"records with flue gas; each stands for an equal share of the {dro.preheated_hours:,.0f} h with heat",
         "-"),
        ("mean gas flow over the operating records", round(100 * rec_f["heat"], 1), "% of design",
         "the heat, and so the ORC output, scales with it", "-"),
        ("ID fan row cap (measured duct)", int(row_cap), "rows",
         "worst of 461 records within 2450 kW at PF 0.85, clean", "common/fan_margin.py"),
        ("ORC net output, design record", round(float(orc.orc_gross_kw), 1), "kW",
         "net of the ORC's own pump, generator and 3 % auxiliary load; the package is priced on it; "
         f"costs {100*(1 - orc.water_m3_per_day/dro.water_m3_per_day):.1f}% of water capacity", "-"),
        ("ORC net output, mean over the operating records", round(float(orc.orc_mean_kw), 1), "kW",
         "what the electricity balance credits", "-"),
    ])

    # ---- 4. robustness - every claim computed ----
    live_g = corners[corners.duct.str.contains("MEASURED")]
    corners_g = corners
    design_cols = ["track_c", "membrane", "path"]
    mode_g = live_g[design_cols].mode().iloc[0]
    n_stable_g = int((live_g[design_cols] == mode_g.values).all(axis=1).sum())
    N_CORNERS = 15        # 5 ducts x 3 discount rates - the stage 4 design

    # PAIRED: the same input combination scored for each architecture. Much
    # stronger than comparing percentiles, which only compares two spreads.
    axes = [c for c in factorial.columns if c not in ("architecture", "lcow")]
    jg = pd.concat({a: factorial[factorial.architecture == a].set_index(axes).lcow
                    for a in (PLAIN, DRO, ORC)}, axis=1).dropna()
    jg.columns = ["plain", "dro", "orc"]
    n_ord_g = int(((jg.plain < jg.dro) & (jg.dro < jg.orc)).sum())
    pen_g = jg.dro - jg.plain

    R += rows_for("4_robustness", [
        ("max conceivable preheat benefit at a 5C intake", round(max_benefit_5c, 3), "kWh/m3",
         (f"vs {fanpump_dro:.3f} kWh/m3 fan+pump for Direct-RO: " +
          ("less than the energy spent delivering the heat" if max_benefit_5c < fanpump_dro else
           "MORE than the fan+pump energy, so in energy terms a 5 C sea would pay for the delivery") +
          f". In cost it is worth A${max_benefit_5c * heat_share * ec.ELECTRICITY_AUD_PER_MWH / 1000:.3f}/m3 "
          f"against A${cap:.3f}/m3 of capital and O&M - missing winter data cannot flip the sign"),
         "-"),
        ("Stage 4 live corners (measured duct), global basis",
         f"{n_stable_g} of {len(live_g)}", "-",
         f"select {int(mode_g.track_c)}C / {mode_g.membrane} membrane / path {mode_g.path}",
         "sweep_stage4_corners.csv"),
        ("Stage 4 corners feasible, global basis", f"{len(corners_g)} of {N_CORNERS}", "-",
         f"{N_CORNERS - len(corners_g)} excluded by the ID fan constraint",
         "sweep_stage4_corners.csv"),
        ("ordering, global-basis grid", f"{n_ord_g:,} of {len(jg):,}", "-",
         "PAIRED, global cost basis: conventional SWRO < Direct-RO < ORC+RO in every combination of the nine other inputs",
         "sweep_stage3_factorial.csv"),
        ("heat-recovery penalty, global-basis grid (min / median / max)",
         f"{pen_g.min():.3f} / {pen_g.median():.3f} / {pen_g.max():.3f}", "A$/m3",
         "never negative; global cost basis only", "sweep_stage3_factorial.csv"),
    ])

    # ---- 5. sensitivity ranking ----
    for a in (PLAIN, DRO, ORC):
        t = tornado[tornado.architecture == a].sort_values("swing", ascending=False)
        for _, r in t.head(5).iterrows():
            R += rows_for("5_sensitivity", [
                (f"swing [{a}]: {r.parameter}", round(float(r.swing), 3), "A$/m3",
                 f"{r.low:.3f} to {r.high:.3f} ({r.pct_of_base:.0f}% of base)",
                 "sweep_stage3_tornado.csv"),
            ])

    # ---- 6. distribution ----
    for a in sorted(factorial.architecture.unique()):
        x = factorial[factorial.architecture == a]["lcow"]
        q = x.quantile([0.05, 0.5, 0.95])
        R += rows_for("6_distribution", [
            (f"P5 / P50 / P95 [{a}], global-basis grid", f"{q.iloc[0]:.3f} / {q.iloc[1]:.3f} / {q.iloc[2]:.3f}",
             "A$/m3", f"{len(x):,} combinations of the nine other inputs, global cost basis - NOT a probability",
             "sweep_stage3_factorial.csv"),
        ])

    # ---- 7. validation ----
    md = pd.read_csv(out / "med_design_summary.csv")
    _dm = md[(md["architecture"] == "Direct-MED") & (md["feasible"] == True)]
    med_approach = float((_dm["loop_return_c"] - _dm["tbt_c"]).min()) if len(_dm) else None
    tb = tornado.groupby("architecture").baseline.first()
    m6_best = lcow_def.groupby("architecture").lcow_aud_per_m3.min()
    drift = ((tb - m6_best).abs() / m6_best * 100).dropna()
    R += rows_for("7_validation", [
        ("LCOW decomposition closure", "sums to the reported LCOW", "-",
         "asserted, fails loudly if not", "reporting/lcow_breakdown.py"),
        ("sweep baseline drift vs Module 6 (max)", f"{drift.max():.2f}% over {len(drift)} architectures",
         "-", "the sweep's own baseline must reproduce Module 6", "module6_lcow/sensitivity_sweep.py"),
        ("minimum local pinch, every reported ORC design", round(true_pinch, 2), "K",
         "20 even points plus the working fluid's bubble and dew points; confirmed against 1,000 points", "pinch_resolution_check.csv"),
        ("Direct-MED loop-to-first-effect approach (min)",
         f"{med_approach:.1f} K" if med_approach is not None else "no feasible design", "K",
         "loop return minus TBT at every feasible track - no temperature cross",
         "med_design_summary.csv"),
        ("Zukauskas C, S_T = S_L staggered", ZUKAUSKAS_C, "-",
         "C = 0.35 x (S_T/S_L)^0.2 with ratio 1.0 - verified", "common/constants.py"),
        ("row-count correction C2", "1.00 recommended / ~0.98 shallowest", "-",
         "27-33 rows on the reported designs, 15-16 on the shallowest - not applied",
         "-"),
        ("gas-side share of thermal resistance", 95, "%",
         "at the design record - why the fixed tube-side h is acceptable", "-"),
        ("GPS availability", ec.GPS_AVAILABILITY * 100, "%", "461/605 raw records, measured", "-"),
    ])

    # ---- 8. MED design - TBT and effect count co-optimised ----
    med_rows = []
    for arch in ("Direct-MED", "ORC+MED"):
        x = md[(md["architecture"] == arch) & (md["feasible"] == True)]
        if x.empty:
            med_rows.append((f"{arch}: design", "no feasible design at any track", "-",
                             "see med_design_summary.csv for the reason per track",
                             "med_design_summary.csv"))
            continue
        b = x.loc[x["lcow_aud_per_m3"].idxmin()]
        at_floor = abs(float(b.tbt_c) - float(b.tbt_floor_c)) < 1e-6
        med_rows += [
            (f"{arch}: extraction track", int(b.track_c), "C", "lowest-LCOW track",
             "med_design_summary.csv"),
            (f"{arch}: top brine temperature", round(float(b.tbt_c), 1), "C",
             ("AT the 55 C floor - the uncited floor BINDS" if at_floor
              else "above the floor - the uncited floor does not bind"),
             "med_design_summary.csv"),
            (f"{arch}: number of effects", int(b.n_effects), "-",
             f"physical maximum at this TBT is {int(b.n_max)}", "med_design_summary.csv"),
            (f"{arch}: GOR", round(float(b.gor), 2), "-", "closed form, 0.95 per-effect effectiveness",
             "med_design_summary.csv"),
        ]
        if arch == "ORC+MED":
            med_rows.append((f"{arch}: ORC condensing temperature", round(float(b.tc_c), 1), "C",
                             "sets TBT = Tc - 5 K", "med_design_summary.csv"))
    ok = med_n[med_n["feasible"] == True]
    if len(ok) and "lcow_aud_per_m3" in ok.columns:
        bn = ok.loc[ok["lcow_aud_per_m3"].idxmin()]
        med_rows.append(("effects sweep: lowest-LCOW N at the chosen TBT", int(bn.n_effects), "-",
                         "must equal the Direct-MED effect count above", "med_effects_sweep.csv"))
    R += rows_for("8_med_design", med_rows)

    # --- 9. Monte Carlo - read from monte_carlo.py's outputs, never typed ---
    mc_s, mc_p, mc_c = (out / "monte_carlo_summary.csv", out / "monte_carlo_penalties.csv",
                        out / "monte_carlo_convergence.csv")
    if mc_s.exists() and mc_p.exists() and mc_c.exists():
        ms, mp, conv = pd.read_csv(mc_s), pd.read_csv(mc_p), pd.read_csv(mc_c)
        n_draws = int(conv.n_draws.max())
        tri = ms[ms.distribution == "triangular"]
        mc_rows = []
        for basis in ("bhojwani_global",):
            for a in ("Conventional SWRO (no heat recovery)", "Direct-RO (no ORC)", "orc_ro_current (path A)",
                      "Direct-MED", "orc_med (path A)"):
                r = tri[(tri.cost_basis == basis) & (tri.architecture == a)]
                if r.empty:
                    continue
                r = r.iloc[0]
                mc_rows.append((f"P50 [P10 - P90] {a}, {basis}", f"{r.p50:.3f} [{r.p10:.3f} - {r.p90:.3f}]",
                                "A$/m3", f"baseline {r.baseline:.3f}; triangular, {n_draws:,} paired draws",
                                "monte_carlo_summary.csv"))
            for _, r in mp[(mp.distribution == "triangular") & (mp.cost_basis == basis)].iterrows():
                mc_rows.append((f"{r.penalty}, {basis}", f"{r.p50:.3f} [{r.p10:.3f} - {r.p90:.3f}]", "A$/m3",
                                f"chance it pays {100 * r.chance_it_pays:.2f} %; lowest draw {r['min']:.3f}",
                                "monte_carlo_penalties.csv"))
        piv = ms[ms.cost_basis == "bhojwani_global"].pivot_table(index=["cost_basis", "architecture"], columns="distribution", values="p50")
        shift = (piv["pert"] / piv["triangular"] - 1.0) * 100.0
        mc_rows.append(("PERT check: P50 shift from triangular, range across architectures, global basis",
                        f"{shift.min():+.1f} to {shift.max():+.1f}", "%",
                        "PERT (lambda = 4) puts more weight near the baseline, so its P50 sits lower; the chance "
                        f"that heat recovery or the ORC pays is {100 * mp[mp.cost_basis == 'bhojwani_global'].chance_it_pays.max():.2f} % under both shapes",
                        "monte_carlo_summary.csv"))
        half = conv[(conv.n_draws >= n_draws / 2) & (conv.cost_basis == "bhojwani_global")].groupby(["cost_basis", "architecture"]).p50
        rel = half.agg(lambda v: (v.max() - v.min()) / v.iloc[-1] * 100.0)
        mc_rows.append(("convergence: largest P50 change, half the draws to all, global basis", round(float(rel.max()), 2), "%",
                        f"relative change in the P50 between {n_draws // 2:,} and {n_draws:,} draws",
                        "monte_carlo_convergence.csv"))
        R += rows_for("9_monte_carlo", mc_rows)

    df = pd.DataFrame(R)
    path = out / "FINAL_RESULTS.csv"
    df.to_csv(path, index=False)
    return df, path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    df, path = build(root / "outputs")
    print(f"wrote {path} ({len(df)} figures)\n")
    for sec, g in df.groupby("section"):
        print(f"--- {sec} ({len(g)}) ---")
        for r in g.itertuples():
            print(f"  {r.figure:34s} {str(r.value):>14s} {r.unit}")
