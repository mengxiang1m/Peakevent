"""
Merge weather data (from Excel) with WLEL load data (from CSV).

Usage:
    python tools/prepare_weather_merge.py \
        --weather_path dataset/load_data/hf_load_data/浦东天气数据20120101-20250930.xlsx \
        --load_path dataset/load_data/hf_load_data/hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
        --output_path dataset/load_data/hf_load_data/hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3_weather.csv \
        --weather_cols "温度(℃),湿度(%)" \
        --rename_cols "temperature,humidity"

The script is extensible: pass different --weather_cols to include more features
(e.g. "温度(℃),湿度(%),风速(m/s),气压(hPa)").
"""

import argparse
import pandas as pd
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description="Merge weather data with WLEL load data")
    parser.add_argument("--weather_path", type=str, required=True,
                        help="Path to the weather Excel file")
    parser.add_argument("--load_path", type=str, required=True,
                        help="Path to the WLEL CSV file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--weather_cols", type=str, default="温度(℃),湿度(%)",
                        help="Comma-separated Chinese column names from the weather Excel")
    parser.add_argument("--rename_cols", type=str, default="temperature,humidity",
                        help="Comma-separated English column names for the output CSV")
    args = parser.parse_args()

    weather_cols = [c.strip() for c in args.weather_cols.split(",")]
    rename_cols = [c.strip() for c in args.rename_cols.split(",")]
    assert len(weather_cols) == len(rename_cols), \
        f"weather_cols ({len(weather_cols)}) and rename_cols ({len(rename_cols)}) must have the same length"

    # ---- 1. Read weather data ----
    print(f"Reading weather data from: {args.weather_path}")
    df_weather = pd.read_excel(args.weather_path)
    print(f"  Weather shape: {df_weather.shape}, columns: {list(df_weather.columns)}")

    # Build datetime from 年月日 + 小时
    df_weather["datetime"] = pd.to_datetime(df_weather["年月日"]) + pd.to_timedelta(df_weather["小时"], unit="h")
    df_weather = df_weather.set_index("datetime")

    # Keep only the requested columns
    for col in weather_cols:
        if col not in df_weather.columns:
            raise ValueError(f"Column '{col}' not found in weather data. Available: {list(df_weather.columns)}")
    df_weather = df_weather[weather_cols].copy()

    # Rename to English
    rename_map = dict(zip(weather_cols, rename_cols))
    df_weather.rename(columns=rename_map, inplace=True)
    print(f"  Selected columns: {list(df_weather.columns)}")
    print(f"  Weather date range: {df_weather.index.min()} ~ {df_weather.index.max()}")

    # ---- 2. Read load data ----
    print(f"\nReading load data from: {args.load_path}")
    df_load = pd.read_csv(args.load_path)
    print(f"  Load shape: {df_load.shape}, columns: {list(df_load.columns)}")

    # Parse the datetime column used for joining
    df_load["_merge_dt"] = pd.to_datetime(df_load["date_60min"])
    print(f"  Load date range: {df_load['_merge_dt'].min()} ~ {df_load['_merge_dt'].max()}")

    # ---- 3. Merge ----
    # De-duplicate weather (some timestamps have duplicates in source)
    df_weather = df_weather[~df_weather.index.duplicated(keep='first')]
    
    # Left join: keep all load rows, attach weather where available
    df_weather_reset = df_weather.reset_index()
    df_merged = df_load.merge(
        df_weather_reset,
        left_on="_merge_dt",
        right_on="datetime",
        how="left"
    )

    # Drop helper columns
    df_merged.drop(columns=["_merge_dt", "datetime"], inplace=True)

    # ---- 4. Handle missing values ----
    for col in rename_cols:
        n_missing = df_merged[col].isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {n_missing} missing values in '{col}', applying interpolation + fill")
            df_merged[col] = df_merged[col].interpolate(method="linear")
            df_merged[col] = df_merged[col].ffill().bfill()

    n_still_missing = df_merged[rename_cols].isna().sum().sum()
    if n_still_missing > 0:
        print(f"  ERROR: {n_still_missing} values still missing after interpolation!")
    else:
        print(f"  All weather values filled successfully.")

    # ---- 5. Save ----
    print(f"\nSaving merged data to: {args.output_path}")
    print(f"  Final shape: {df_merged.shape}, columns: {list(df_merged.columns)}")
    df_merged.to_csv(args.output_path, index=False)

    # Quick sanity check
    print(f"\n  First 3 rows of new columns:")
    print(df_merged[rename_cols].head(3).to_string())
    print(f"\n  Stats of new columns:")
    print(df_merged[rename_cols].describe().to_string())
    print("\nDone!")


if __name__ == "__main__":
    main()
