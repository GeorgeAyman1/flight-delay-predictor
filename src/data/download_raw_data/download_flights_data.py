import requests
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(os.getenv("PROJECT_ROOT"))

URL = "https://huggingface.co/datasets/YousefSawy/airline_dataset/resolve/main/flights_combined.parquet"
DESTINATION_DIR = ROOT / "data" / "interim" / "flights"
DESTINATION_FILE = DESTINATION_DIR / "flights_combined.parquet"


def download_flights_data(force: bool = False) -> Path:
    """
    Downloads the flight parquet file from Hugging Face.
    :param force: If True, overwrites existing file.
    """
    # 1. Ensure directory exists
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Skip if already exists
    if DESTINATION_FILE.exists() and not force:
        print(f"  Skipping download: {DESTINATION_FILE.name} already exists.")
        return DESTINATION_FILE

    print("Downloading flights data from Hugging Face...")

    try:
        with requests.get(URL, stream=True, timeout=60) as r:
            r.raise_for_status()

            with open(DESTINATION_FILE, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Successfully saved to: {DESTINATION_FILE}")

    except Exception as e:
        print(f"Failed to download flights data: {e}")
        if DESTINATION_FILE.exists():
            DESTINATION_FILE.unlink()
        raise e

    return DESTINATION_FILE


def main():
    print("=" * 60)
    print("download_flights.py")
    print("=" * 60)

    download_flights_data()

    print("\nDone.")
    print("=" * 60)


if __name__ == "__main__":
    main()
