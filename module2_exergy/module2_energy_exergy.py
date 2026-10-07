"""
Module 2: the energy and exergy content of the flue gas, and of the band
recovered at each extraction track, per operating record, against a 25 C
dead state.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.constants import T_AMBIENT_CELSIUS, TARGET_TEMPERATURES

t0_celsius = T_AMBIENT_CELSIUS
# every track, matching Module 1
TRACKS = tuple(sorted(TARGET_TEMPERATURES, reverse=True))


def run_module2(input_csv, output_csv):
    """
    Energy and exergy CONTENT of the recovered band, per row and per track.

    NOTE ON NAMING: recovered_exergy_fraction is the exergy *content* of the
    recovered band relative to the exergy entering - an idealised,
    reversible-recovery upper bound. It is NOT the Gouy-Stodola exergy
    *destruction*, which module2b_exergy_destruction.py computes from the
    actual Module 3 states. Named so it is not mistaken for destruction.

    Exergy here is thermal (physical) only: chemical exergy and the gas's
    small pressure exergy (~-4 kPa gauge) are excluded, which is standard for
    a waste-heat availability analysis. Water vapour stays above its dew point
    at every track, so there is no latent term.
    """
    t0_k = t0_celsius + 273.15
    print(f"loading module 1 outputs from {input_csv}...")
    df = pd.read_csv(input_csv)

    out = {"total_energy_rejected_kw": [], "total_exergy_entering_kw": []}
    for t in TRACKS:
        for k in (f"first_law_efficiency_{t:.0f}_pct", f"recovered_exergy_{t:.0f}_kw",
                  f"recovered_exergy_fraction_{t:.0f}_pct"):
            out[k] = []

    for idx, row in df.iterrows():
        t_in = row['FF_Outlet_Temp_Avg_C']
        m_dot_fg = row['Total gas flow kg/s']
        cp_dead_state = row['dynamic_cp_dead_state']

        if m_dot_fg <= 0 or t_in <= t0_celsius or cp_dead_state <= 0:
            for k in out:
                out[k].append(0.0)
            continue

        t_in_k = t_in + 273.15
        q_rej = m_dot_fg * cp_dead_state * (t_in - t0_celsius)
        ex_entering = m_dot_fg * cp_dead_state * ((t_in - t0_celsius) - t0_k * np.log(t_in_k / t0_k))
        out["total_energy_rejected_kw"].append(q_rej)
        out["total_exergy_entering_kw"].append(ex_entering)

        for t in TRACKS:
            q_rec = row[f'heat_recovered_{t:.0f}_kw']
            cp_t = row[f'dynamic_cp_{t:.0f}']
            t_k = t + 273.15
            ex_rec = (m_dot_fg * cp_t * ((t_in - t) - t0_k * np.log(t_in_k / t_k))) if t_in > t else 0.0
            out[f"first_law_efficiency_{t:.0f}_pct"].append((q_rec / q_rej) * 100.0 if q_rej > 0 else 0.0)
            out[f"recovered_exergy_{t:.0f}_kw"].append(ex_rec)
            out[f"recovered_exergy_fraction_{t:.0f}_pct"].append(
                (ex_rec / ex_entering) * 100.0 if ex_entering > 0 else 0.0)

    for k, v in out.items():
        df[k] = v

    # per-track statistics over the records that recover heat at that track
    print("thermodynamic performance logs:")
    print(f"mean flue gas entering energy rate: {df['total_energy_rejected_kw'].mean():.2f} kw")
    print(f"mean flue gas entering exergy rate: {df['total_exergy_entering_kw'].mean():.2f} kw")
    for t in TRACKS:
        act = df[df[f'heat_recovered_{t:.0f}_kw'] > 0]
        print(f"--- {t:.0f} C track ({len(act)} active rows) ---")
        print(f"  mean first-law recovery: {act[f'first_law_efficiency_{t:.0f}_pct'].mean():.2f}%")
        print(f"  mean recovered exergy: {act[f'recovered_exergy_{t:.0f}_kw'].mean():.2f} kw")
        print(f"  mean recovered exergy fraction: {act[f'recovered_exergy_fraction_{t:.0f}_pct'].mean():.2f}%")

    df.to_csv(output_csv, index=False)
    print(f"module 2 output exported to: {output_csv}\n")


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    run_module2(str(output_dir / "GPS_Module1_Output.csv"), str(output_dir / "GPS_Module2_Output.csv"))
