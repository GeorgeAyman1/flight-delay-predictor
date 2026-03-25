from pathlib import Path
import time
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

STATIONS = [
    "BOS",
    "CLT",
    "DEN",
    "DFW",
    "DTW",
    "EWR",
    "IAH",
    "JFK",
    "MSP",
    "LAX",
    "MIA",
    "ORD",
    "PHX",
    "SEA",
    "SFO",
]

YEARS = [2022, 2023, 2024, 2025]

AIRPORT_TIMEZONES = {
    "BOS": "America/New_York",
    "CLT": "America/New_York",
    "DEN": "America/Denver",
    "DFW": "America/Chicago",
    "DTW": "America/New_York",
    "EWR": "America/New_York",
    "IAH": "America/Chicago",
    "JFK": "America/New_York",
    "MSP": "America/Chicago",
    "LAX": "America/Los_Angeles",
    "MIA": "America/New_York",
    "ORD": "America/Chicago",
    "PHX": "America/Phoenix",
    "SEA": "America/Los_Angeles",
    "SFO": "America/Los_Angeles",
}

WEATHER_FIELDS = [
    "tmpf",  # temperature (F)
    "dwpf",  # dew point (F)
    "relh",  # relative humidity (%)
    "drct",  # wind direction
    "sknt",  # wind speed (knots)
    "gust",  # wind gust (knots)
    "p01i",  # precipitation over previous hour (inches)
    "alti",  # altimeter
    "mslp",  # mean sea level pressure
    "vsby",  # visibility
    "feel",  # apparent temperature
    "wxcodes",  # weather codes
    "skyc1",
    "skyc2",
    "skyc3",
    "skyc4",
    "skyl1",
    "skyl2",
    "skyl3",
    "skyl4",
]

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OUT_DIR = Path("data/raw/weather")
META_PATH = OUT_DIR / "airport_timezone_mapping.csv"


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def build_params(station: str, year: int) -> list[tuple[str, str]]:
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year + 1}-01-01T00:00:00Z"

    params = [
        ("sts", start),
        ("ets", end),
        ("tz", "UTC"),
        ("format", "onlycomma"),
        ("report_type", "3"),
        ("latlon", "yes"),
        ("elev", "yes"),
        ("missing", "empty"),
        ("trace", "empty"),
        ("station", station),
    ]

    for field in WEATHER_FIELDS:
        params.append(("data", field))

    return params


def save_timezone_mapping() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame(
        [
            {"origin_airport": airport, "airport_timezone": tz}
            for airport, tz in AIRPORT_TIMEZONES.items()
        ]
    ).sort_values("origin_airport")
    meta.to_csv(META_PATH, index=False)
    print(f"Saved timezone mapping: {META_PATH}")


def download_one(session: requests.Session, station: str, year: int) -> None:
    out_path = OUT_DIR / f"{station}_{year}_weather_utc.csv"

    if out_path.exists():
        print(f"Skipping existing file: {out_path.name}")
        return

    params = build_params(station, year)
    print(f"Downloading {station} {year} ...")

    response = session.get(IEM_URL, params=params, timeout=300)
    response.raise_for_status()

    text = response.text.strip()
    if not text:
        raise ValueError(f"Empty response for {station} {year}")

    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"Saved: {out_path.name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_timezone_mapping()

    session = build_session()

    failures: list[tuple[str, int, str]] = []

    for station in STATIONS:
        for year in YEARS:
            try:
                download_one(session, station, year)
                time.sleep(1)
            except Exception as e:
                failures.append((station, year, str(e)))
                print(f"FAILED: {station} {year} -> {e}")

    print("\nDownload complete.")

    if failures:
        print("\nFailures:")
        for station, year, error in failures:
            print(f"- {station} {year}: {error}")
    else:
        print("All files downloaded successfully.")


if __name__ == "__main__":
    main()
