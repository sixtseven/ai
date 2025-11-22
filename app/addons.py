import requests
import json
import logging

# Use the same logger name to ensure logs appear in the FastAPI console
logger = logging.getLogger("uvicorn.error")

def fetch_and_save_addons(booking_id: str):
    """
    Fetches addons from the API and saves them to addons.json.
    """
    VEHICLE_API_BASE = "https://hackatum25.sixt.io/"
    url = f"{VEHICLE_API_BASE}/api/booking/{booking_id}/addons"
    
    try:
        resp = requests.get(url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            return data
        else:
            logger.warning(f"Addons API returned {resp.status_code} for {booking_id}")
    except Exception as e:
        logger.error(f"Failed to fetch addons: {e}")

    
