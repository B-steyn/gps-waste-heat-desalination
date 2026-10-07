"""
Module 0: cleans the station's raw spreadsheets into the two CSVs every
other module reads (data/GPS_Operating_Data_Clean.csv and
data/GPS_Water_related_data.csv).
"""

from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "raw_data"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def clean_gps_data(input_file, output_file):
    print(f"Reading {input_file}...")
    # Load raw Excel sheet
    df = pd.read_excel(input_file, sheet_name='Flue Gas')

    # Extract row 0 to use as actual headers
    headers = df.iloc[0].tolist()
    headers[0] = 'Interval'
    headers[1] = 'Timestamp'

    # Drop row 0 from the data rows and assign clean column names
    df_clean = df.iloc[1:].copy()
    df_clean.columns = headers

    # Convert numeric columns to float types safely
    numeric_cols = df_clean.columns[2:]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Mean of the A- and B-side fabric filter outlet temperatures: the gas
    # temperature entering the flue-gas exchanger, which sits just downstream
    df_clean['FF_Outlet_Temp_Avg_C'] = (
                                               df_clean['Fabric Filter Outlet temp © A side '] +
                                               df_clean['Fabric Filter Outlet temp © B side ']
                                       ) / 2.0

    # Count rows before filtering
    total_rows = len(df_clean)

    # Drop idle and shutdown records (unit load below 1 MW)
    df_operating = df_clean[df_clean['Unit Load (MW)'] >= 1.0].copy()
    df_idle = df_clean[df_clean['Unit Load (MW)'] < 1.0].copy()

    idle_count = len(df_idle)
    operating_count = len(df_operating)
    availability_factor = (operating_count / total_rows) * 100

    print("\n DATA CLEANING SUMMARY")
    print(f"Total Logged Records: {total_rows}")
    print(f"Idle/Shutdown Records Removed: {idle_count}")
    print(f"Active Operating Records Retained: {operating_count}")
    print(f"Calculated Plant Availability Factor: {availability_factor:.2f}%")

    # Save the operating data to a clean intermediate CSV file, preserving
    # full float precision and ISO timestamps (no Excel round-trip)
    df_operating.to_csv(output_file, index=False)
    print(f"Cleaned operating dataset successfully saved to: {output_file}\n")

    return df_operating


def clean_seawater_data(input_file, output_file):
    """
    Cleans the seawater intake temperature data. The source carries full
    6-hourly timestamps, matching the flue-gas records; writing the CSV
    straight from it keeps the time of day, so Modules 3 and 4a can merge
    the two datasets on an exact 6-hourly key.
    """
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file)
    df.columns = df.columns.str.strip()

    # drop the empty index-artifact column if present
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    df.to_csv(output_file, index=False)
    print(f"Cleaned seawater dataset successfully saved to: {output_file}\n")
    return df


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    clean_gps_data(
        str(RAW_DATA_DIR / "GPS_Data_received_v2.xlsx"),
        str(DATA_DIR / "GPS_Operating_Data_Clean.csv"),
    )
    clean_seawater_data(
        str(RAW_DATA_DIR / "GPS_Water_related_data.xlsx"),
        str(DATA_DIR / "GPS_Water_related_data.csv"),
    )
