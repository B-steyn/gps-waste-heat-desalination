"""
Monte Carlo analysis of LCOW.

The stage 3 grid answers "can any combination of inputs flip the result?"
It cannot answer "what is the most probable LCOW?", because a grid weights
every combination equally, pessimistic corners included. Here each swept
economic input is drawn from a TRIANGULAR distribution whose minimum, most
likely and maximum values are the grid's own low, baseline and high (PERT
is run once as a check). It reports the P50 - an even chance of landing
above or below, the cost-estimating "most probable outcome" - with P10-P90.

Rules:
  - Every architecture is scored on the SAME draws, so the penalties are
    paired and the chance that heat recovery pays is read off directly.
  - The Lang factor and Bhojwani's cost error stay separate inputs, as in
    the grid (installation scope against equipment price). This is stated,
    not hidden.
  - Inputs are drawn independently; no correlations are assumed.
  - Designs are held at their baseline optimum, as in the grid (stage 4
    shows the design does not move at the corners).
  - The S-curves give the probability that LCOW falls below ANY price, so a
    reader holding a water tariff can read off the chance of beating it.

Scoring reuses sensitivity_sweep.evaluate() unchanged, and a draw at the
baseline must reproduce Module 6 (asserted).

Outputs (outputs/):
  monte_carlo_summary.csv      mean and P5/P10/P50/P90/P95 per architecture, both shapes
  monte_carlo_penalties.csv    the paired heat-recovery and ORC penalties, same statistics
  monte_carlo_scurves.csv      LCOW at every 0.5 % of probability - the S-curves
  monte_carlo_convergence.csv  P50 on the first 5,000 / 10,000 / 20,000 / 40,000 draws
  monte_carlo_draws.csv        every triangular draw: its inputs and each architecture's LCOW
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import sensitivity_sweep as sweep

OUT = ROOT / "outputs"
N_DRAWS = 40_000
SEED = 20260915
BASES = ("bhojwani_global",)
PLAIN, DRO, ORC = "Conventional SWRO (no heat recovery)", "Direct-RO (no ORC)", "orc_ro_current (path A)"
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def inputs():
    """(label, attribute, minimum, most likely, maximum) - the grid's own axes, never retyped."""
    return [(label, attr, min(values), base, max(values))
            for label, attr, values, base in sweep.sweep_axes()]


def draw(n, shape, seed=SEED):
    """n independent draws of every input. PERT uses the standard lambda = 4."""
    rng = np.random.default_rng(seed)
    cols = {}
    for _, attr, lo, mode, hi in inputs():
        if shape == "triangular":
            cols[attr] = rng.triangular(lo, mode, hi, n)
        elif shape == "pert":
            a = 1.0 + 4.0 * (mode - lo) / (hi - lo)
            b = 1.0 + 4.0 * (hi - mode) / (hi - lo)
            cols[attr] = lo + (hi - lo) * rng.beta(a, b, n)
        else:
            raise ValueError(shape)
    return pd.DataFrame(cols)


def score(designs, basis, X):
    """LCOW for every draw and every architecture, on one cost basis - paired by construction."""
    rows = X.to_dict("records")
    return pd.DataFrame({d["architecture"]: [sweep.evaluate(d, anchor=basis, **r) for r in rows]
                         for d in designs})


def stats(s):
    q = s.quantile(QUANTILES)
    return {"mean": float(s.mean()), **{f"p{round(p * 100):02d}": float(q[p]) for p in QUANTILES}}


def main():
    designs = sweep.load_designs(OUT)
    base_kw = {attr: mode for _, attr, _, mode, _ in inputs()}
    baseline = {}
    for basis in BASES:
        m6 = pd.read_csv(OUT / f"gps_module6_lcow_{basis}_lcow.csv")
        for d in designs:
            b = sweep.evaluate(d, anchor=basis, **base_kw)
            row = m6[(m6.architecture == d["architecture"])
                     & ((m6.track_c == d["track_c"]) | (m6.track_c.isna() & pd.isna(d["track_c"])))]
            ref = float(row.lcow_aud_per_m3.iloc[0])
            assert abs(b - ref) / ref < 1e-3, f"{basis} {d['architecture']}: baseline {b:.4f} vs Module 6 {ref:.4f}"
            baseline[(basis, d["architecture"])] = b
    print(f"baseline reproduces Module 6 for {len(designs)} architectures")

    summary, penalties, curves, conv, draws_out = [], [], [], [], []
    probs = np.round(np.arange(0.0, 1.0001, 0.005), 3)
    for shape in ("triangular", "pert"):
        X = draw(N_DRAWS, shape)
        for basis in BASES:
            L = score(designs, basis, X)
            for a in L.columns:
                summary.append({"distribution": shape, "cost_basis": basis, "architecture": a,
                                "baseline": baseline[(basis, a)], **stats(L[a]),
                                "share_of_draws_below_baseline": float((L[a] < baseline[(basis, a)]).mean())})
            for name, s in (("heat recovery: Direct-RO - conventional SWRO", L[DRO] - L[PLAIN]),
                            ("the ORC: ORC+RO - Direct-RO", L[ORC] - L[DRO])):
                penalties.append({"distribution": shape, "cost_basis": basis, "penalty": name, **stats(s),
                                  "min": float(s.min()), "max": float(s.max()),
                                  "chance_it_pays": float((s < 0).mean())})
            if shape == "triangular":
                for a in L.columns:
                    curves += [{"cost_basis": basis, "architecture": a, "probability": p, "lcow_aud_per_m3": v}
                               for p, v in zip(probs, L[a].quantile(probs).values)]
                    for n in (5_000, 10_000, 20_000, 40_000):
                        conv.append({"cost_basis": basis, "architecture": a, "n_draws": n,
                                     "p50": float(L[a].iloc[:n].median())})
                draws_out.append(pd.concat([pd.Series(basis, index=X.index, name="cost_basis"),
                                            X.round(6), L], axis=1))

    summary, penalties = pd.DataFrame(summary), pd.DataFrame(penalties)
    summary.to_csv(OUT / "monte_carlo_summary.csv", index=False)
    penalties.to_csv(OUT / "monte_carlo_penalties.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "monte_carlo_scurves.csv", index=False)
    pd.DataFrame(conv).to_csv(OUT / "monte_carlo_convergence.csv", index=False)
    pd.concat(draws_out, ignore_index=True).to_csv(OUT / "monte_carlo_draws.csv", index=False,
                                                   float_format="%.6g")

    for shape in ("triangular", "pert"):
        print(f"\n== {shape}: P50 [P10 - P90] against the baseline, A$/m3")
        for basis in BASES:
            s = summary[(summary.distribution == shape) & (summary.cost_basis == basis)]
            print(f"  {basis}")
            for r in s.itertuples():
                print(f"    {r.architecture:32s} {r.p50:7.3f} [{r.p10:6.3f} - {r.p90:6.3f}]   baseline {r.baseline:6.3f}"
                      f"   {100 * r.share_of_draws_below_baseline:4.1f} % of draws below it")
        print("  paired penalties:")
        for r in penalties[penalties.distribution == shape].itertuples():
            print(f"    {r.cost_basis:15s} {r.penalty:40s} P50 {r.p50:6.3f} [{r.p10:6.3f} - {r.p90:6.3f}]"
                  f"  min {r.min:6.3f}  chance it pays {100 * r.chance_it_pays:.2f} %")
    c = pd.DataFrame(conv)
    spread = c.groupby(["cost_basis", "architecture"]).p50.agg(lambda v: v.iloc[-2:].max() - v.iloc[-2:].min())
    print(f"\nconvergence: largest P50 change from 20,000 to 40,000 draws = {spread.max():.4f} A$/m3")


if __name__ == "__main__":
    main()
