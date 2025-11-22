import requests
import json
import logging

# Use the same logger name to ensure logs appear in the FastAPI console
logger = logging.getLogger("uvicorn.error")

def fetch_and_save_addons():
    """
    Fetches addons from the API and saves them to addons.json.
    """
    booking_id = "Q80chI4GLq4d4D3UR9p5"
    VEHICLE_API_BASE = "https://hackatum25.sixt.io/"
    url = f"{VEHICLE_API_BASE}/api/booking/{booking_id}/addons"
    
    try:
        resp = requests.get(url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            # Save to local file
            with open("addons.json", "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved addons for {booking_id} to addons.json")
        else:
            logger.warning(f"Addons API returned {resp.status_code} for {booking_id}")
    except Exception as e:
        logger.error(f"Failed to fetch addons: {e}")

if __name__ == "__main__":
    fetch_and_save_addons()