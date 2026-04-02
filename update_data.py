"""
Solar Resource API  - monthly averages (GHI, DNI, tilt) for a location
Docs: https://developer.nlr.gov/docs/solar/
"""

import requests
import pandas as pd
import io
import json
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────────────
API_KEY   = "tTqUkXsER8iTWUXVDzM2KqHLpCgas7xkcodR4TXD"   
BASE_URL  = "https://developer.nrel.gov"

#Get Hourly Data
def get_hourly_nsrdb_timeseries(
    lat:         float,
    lon:         float,
    year:        str  = "2024",
    interval:    int  = 60,
    email:       str  = "ehaenel3@gatech.edu",
    attributes:  Optional[str] = None,
    leap_day:    bool = False,
    utc:         bool = False,
) -> pd.DataFrame:

    if attributes is None:
        attributes = (
            "ghi,dhi,dni,wind_speed,air_temperature,"
            "solar_zenith_angle,surface_pressure,relative_humidity"
        )

    url = f"{BASE_URL}/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"

    params = {
        "api_key":      API_KEY,
        "wkt":          f"POINT({lon} {lat})",
        "names":        year,
        "interval":     interval,
        "attributes":   attributes,
        "email":        email,
        "leap_day":     str(leap_day).lower(),
        "utc":          str(utc).lower(),
        "full_name":    "Solar Data User",
        "affiliation":  "Independent",
        "reason":       "Research",
        "mailing_list": "false",
    }

    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()

    lines = resp.text.splitlines()

    # ── Parse metadata from rows 0 and 1 ──────────────────────────────────────
    meta_keys   = lines[0].split(",")
    meta_values = lines[1].split(",")
    meta = dict(zip(meta_keys, meta_values))

    # ── Parse data ────────────────────────────────────────────────────────────
    df = pd.read_csv(io.StringIO("\n".join(lines[2:])))

    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df = df.set_index("datetime")
    df.drop(['Hour', 'Month', 'Day', 'Minute', 'Year'], axis=1, inplace=True)

    # ── Attach metadata as columns ────────────────────────────────────────────
    desired_meta = {
        "Location_ID": meta.get("Location ID"),
        "Elevation":   meta.get("Elevation"),
    }
    for col, val in desired_meta.items():
        df[col] = val

    print(f"[NSRDB] {meta.get('City')}, {meta.get('State')} | Elev: {meta.get('Elevation')}m")
    print(f"[NSRDB] Retrieved {len(df)} rows")

    return df

#Get Daily Data (total)
def get_daily_nsrdb_timeseries(
    lat:         float,
    lon:         float,
    year:        str  = "2024",
    interval:    int  = 60,
    email:       str  = "ehaenel3@gatech.edu",
    attributes:  Optional[str] = None,
    leap_day:    bool = False,
    utc:         bool = False,
) -> pd.DataFrame:

    if attributes is None:
        attributes = (
            "ghi,dhi,dni,wind_speed,air_temperature,"
            "solar_zenith_angle,surface_pressure,relative_humidity"
        )

    url = f"{BASE_URL}/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"

    params = {
        "api_key":      API_KEY,
        "wkt":          f"POINT({lon} {lat})",
        "names":        year,
        "interval":     interval,
        "attributes":   attributes,
        "email":        email,
        "leap_day":     str(leap_day).lower(),
        "utc":          str(utc).lower(),
        "full_name":    "Solar Data User",
        "affiliation":  "Independent",
        "reason":       "Research",
        "mailing_list": "false",
    }

    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()

    lines = resp.text.splitlines()

    # ── Parse metadata from rows 0 and 1 ──────────────────────────────────────
    meta_keys   = lines[0].split(",")
    meta_values = lines[1].split(",")
    meta = dict(zip(meta_keys, meta_values))

    # ── Parse data ────────────────────────────────────────────────────────────
    df = pd.read_csv(io.StringIO("\n".join(lines[2:])))

    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df = df.set_index("datetime")
    df.drop(['Hour', 'Month', 'Day', 'Minute', 'Year'], axis=1, inplace=True)

    # ── Aggregate to daily ────────────────────────────────────────────────────
    sum_cols  = [c for c in ["GHI", "DHI", "DNI"] if c in df.columns]
    mean_cols = [c for c in ["Wind Speed", "Temperature", "Solar Zenith Angle",
                              "Pressure", "Relative Humidity"] if c in df.columns]
    agg = {}
    for c in sum_cols:
        agg[c] = pd.NamedAgg(column=c, aggfunc="sum")
    for c in mean_cols:
        agg[c] = pd.NamedAgg(column=c, aggfunc="mean")

    df = df.resample("D").agg(**agg)

    # ── Attach metadata as columns ────────────────────────────────────────────
    desired_meta = {
        "Location_ID": meta.get("Location ID"),
        "Elevation":   meta.get("Elevation"),
    }
    for col, val in desired_meta.items():
        df[col] = val

    print(f"[NSRDB] {meta.get('City')}, {meta.get('State')} | Elev: {meta.get('Elevation')}m")
    print(f"[NSRDB] Retrieved {len(df)} rows")

    return df

if __name__ == "__main__":

    cities = [
        ('Phoenix',       33.4484,  -112.0740),   # Desert Southwest
        ('Los Angeles',   34.0522,  -118.2437),   # Pacific Coast
        ('Atlanta',       33.7490,   -84.3880),   # Southeast
        ('Chicago',       41.8781,   -87.6298),   # Midwest
        ('Boston',        42.3601,   -71.0589),   # Northeast
        ('Denver',        39.7392,  -104.9903),   # Mountain
        ('Seattle',       47.6062,  -122.3321),   # Pacific Northwest
    ]

    years = ["2021", "2022", "2023", "2024"]
    
    # --- Pull & cache data --------------------------------
    frames = []
    for name, lat, lon in cities:
        for year in years:
            print(f"Pulling {name} {year}...")
            df = get_daily_nsrdb_timeseries(lat=lat, lon=lon, year=year)
            df["City"] = name
            frames.append(df)
    irradiance_df = pd.concat(frames)
    irradiance_df.to_parquet("irradiance_2021_2024.parquet")
    
    # --- Load from cache -------------------------------------------------------
    irradiance_df = pd.read_parquet("irradiance_2021_2024.parquet")

    print("\n=== NSRDB Daily Time-Series (2021-2024) ===")
    irradiance_df.head()
    #GHI is y variable 