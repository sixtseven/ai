from typing import Any, Dict, List, Optional
import os
import logging
import requests
import json
import time
import random
import re
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Vehicle Upsell Recommender")

# --- LOGGING CONFIGURATION ---
LOG_FILE = os.environ.get("LOG_FILE", "out.txt")
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("uvicorn.error")
logger.propagate = False

# --- CLASS DEFINITIONS ---
class Vehicle(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    price: Optional[float] = None
    seats: Optional[int] = None
    luggage: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None

# --- HELPER FUNCTIONS ---

def get_features_for_booking(booking_id: str) -> Dict[str, int]:
    """Mocked features provider."""
    try:
        v = abs(hash(booking_id))
    except Exception:
        v = 42
    people = 2 + (v % 4)  # 2..5
    luggages = 0 + (v % 3)  # 0..2
    return {"numberOfPeople": int(people), "numberOfLuggages": int(luggages)}

def _extract_vehicle_fields(raw: Dict[str, Any]) -> Vehicle:
    """Parses raw vehicle/deal data into a structured Vehicle object."""
    deal = raw
    vehicle_raw = raw
    pricing = None
    
    if isinstance(raw, dict) and "vehicle" in raw:
        vehicle_raw = raw.get("vehicle") or {}
        pricing = raw.get("pricing") or {}

    # IDs and Names
    vid = None
    if isinstance(vehicle_raw, dict):
        vid = vehicle_raw.get("id") or vehicle_raw.get("vehicleId")
    if not vid:
        vid = raw.get("id") or raw.get("vehicleId") or raw.get("vehicle_type_id")

    name = None
    if isinstance(vehicle_raw, dict):
        brand = vehicle_raw.get("brand") or ""
        model = vehicle_raw.get("model") or ""
        name = (brand + " " + model).strip() or None
    if not name:
        name = raw.get("name") or raw.get("vehicleType") or raw.get("label")

    # Price Logic
    price = None
    try:
        if isinstance(pricing, dict):
            tp = pricing.get("totalPrice")
            if isinstance(tp, dict) and "amount" in tp:
                price = float(tp["amount"])
            elif "displayPrice" in pricing and isinstance(pricing["displayPrice"], dict) and "amount" in pricing["displayPrice"]:
                price = float(pricing["displayPrice"]["amount"])
    except Exception:
        price = None

    if price is None:
        if isinstance(vehicle_raw, dict) and "vehicleCost" in vehicle_raw and isinstance(vehicle_raw["vehicleCost"], dict) and "value" in vehicle_raw["vehicleCost"]:
            try:
                price = float(vehicle_raw["vehicleCost"]["value"]) / 100.0 if vehicle_raw["vehicleCost"]["value"] > 1000 else float(vehicle_raw["vehicleCost"]["value"])
            except Exception:
                price = None
        else:
            for k in ("price", "priceTotal", "amount", "price_eur", "price_cents"):
                if k in raw:
                    try:
                        price = float(raw[k])
                        break
                    except Exception:
                        pass

    # Seats
    seats = None
    if isinstance(vehicle_raw, dict):
        seats = vehicle_raw.get("passengersCount") or vehicle_raw.get("seats")
        if seats is not None:
            try:
                seats = int(seats)
            except Exception:
                seats = None
    if seats is None:
        for k in ("seats", "seat_count", "capacity"):
            if k in raw:
                try:
                    seats = int(raw[k])
                    break
                except Exception:
                    pass

    # Luggage
    luggage = None
    if isinstance(vehicle_raw, dict):
        luggage = vehicle_raw.get("bagsCount") or vehicle_raw.get("luggage")
        if luggage is not None:
            try:
                luggage = int(luggage)
            except Exception:
                luggage = None
    if luggage is None:
        attrs = []
        if isinstance(vehicle_raw, dict) and isinstance(vehicle_raw.get("attributes"), list):
            attrs = vehicle_raw.get("attributes")
        elif isinstance(raw, dict) and isinstance(raw.get("attributes"), list):
            attrs = raw.get("attributes")
        for a in attrs:
            try:
                title = (a.get("title") or "").lower()
                val = a.get("value")
                if "trunk" in title or "boot" in title or "bags" in title:
                    luggage = int(val)
                    break
            except Exception:
                continue

    return Vehicle(id=str(vid) if vid is not None else None, name=name, price=price, seats=seats, luggage=luggage, raw=deal)

def choose_best_upsell(base: Vehicle, candidates: List[Vehicle], people: int, luggages: int) -> Dict[str, Any]:
    """Selects the best upgrade option based on needs and price."""
    best = candidates[0] if candidates else None
    best_score = None
    base_price = base.price or 0.0

    for v in candidates:
        price = v.price or float("inf")
        seats = v.seats if v.seats is not None else 0
        lug = v.luggage if v.luggage is not None else 0

        people_short = max(0, people - seats)
        lug_short = max(0, luggages - lug)
        price_diff = max(0.0, price - base_price)

        # Score: prefer zero shortages, then minimal price difference, then higher seat count
        score = (people_short + lug_short, price_diff, -seats)

        if best_score is None or score < best_score:
            best_score = score
            best = v

    reason = {
        "people": people,
        "luggages": luggages,
        "base_price": base_price,
        "chosen_score": best_score,
    }
    return {"vehicle": best, "reason": reason}

# -----------------------------------------------------------
# --- GENERATE UPSELL REASONS (GEMINI EXCLUSIVE) ---
# -----------------------------------------------------------

def generate_upsell_reasons(base: Vehicle, upsell: Optional[Vehicle], people: int, luggages: int) -> List[str]:
    """
    Generate up to 3 personalized, human-readable reasons to upsell using Gemini.
    Falls back to a rule-based generator if Gemini fails or API key is missing.
    """
    if upsell is None:
        return []

    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    # 1. Attempt Gemini Generation
    if gemini_key:
        try:
            return _call_gemini_api(gemini_key, base, upsell, people, luggages)
        except Exception as e:
            logger.error(f"Gemini generation failed, falling back to rules: {e}")
            # Proceed to rule-based fallback below
    else:
        logger.warning("GEMINI_API_KEY not found. Using rule-based fallback.")

    # 2. Rule-Based Fallback (if Gemini fails or is missing)
    return _generate_rules_based_reasons(base, upsell, people, luggages)


def _call_gemini_api(api_key: str, base: Vehicle, upsell: Vehicle, people: int, luggages: int) -> List[str]:
    """Internal function to handle the HTTP request to Gemini."""
    
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    base_path = f"https://generativelanguage.googleapis.com/v1/models/{gemini_model}:generateContent"
    
    # --- UPDATED PROMPT LOGIC ---
    system_msg = (
        "You are a persuasive automotive sales copywriter. You are recommending a premium vehicle upgrade to a customer. "
        "Your goal is to provide convincing arguments why the user should buy the more expensive car based on their specific needs.\n\n"
        "GUIDELINES:\n"
        "1. **NEVER use the word 'upsell'** in the output. Use words like 'upgrade', 'premium', 'spacious', or 'comfort'.\n"
        "2. Focus on specific benefits: interior space for the specific number of passengers, trunk capacity for their luggage, and premium features (speed, comfort, technology).\n"
        "3. Tone: Enthusiastic, professional, and convincing.\n"
        "4. Format: Return ONLY a JSON array of exactly three strings."
    )

    user_msg = (
        f"Context: The customer is traveling with {people} people and {luggages} pieces of luggage.\n"
        f"Current Base Car: {base.name} (Price: {base.price})\n"
        f"Recommended Upgrade: {upsell.name} (Price: {upsell.price}, Seats: {upsell.seats}, Luggage Capacity: {upsell.luggage})\n\n"
        "Task: Write 3 persuasive bullet points convincing the customer to choose the Upgrade. "
        "Highlight spacious interior, trunk space, and premium comfort. "
        "Output JSON array only."
    )
    # ----------------------------

    body = {
        "contents": [{
            "parts": [{"text": system_msg + "\n\n" + user_msg}], 
            "role": "user"
        }],
        "config": {
            "temperature": float(os.environ.get("GEMINI_TEMPERATURE", "0.7"))
        }
    }

    # API Call with Retries
    max_attempts = int(os.environ.get("GEMINI_MAX_RETRIES", "3"))
    url = f"{base_path}?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    resp = None
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=10.0)
            if resp.status_code == 200:
                break
            if resp.status_code == 429: # Rate limit
                time.sleep(2 ** attempt) 
                continue
            if 400 <= resp.status_code < 500: # Client error, do not retry
                break
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.0)
            
    if resp is None or resp.status_code != 200:
        error_detail = resp.text if resp else str(last_err)
        raise Exception(f"Gemini API Error: {error_detail}")

    # Parse Response
    try:
        j = resp.json()
        # Navigate standard Gemini response structure
        text = j["candidates"][0]["content"]["parts"][0]["text"]
        
        # Regex extract JSON list
        m = re.search(r"\[.*\]", str(text), flags=re.S)
        if not m:
            raise ValueError("No JSON array found in Gemini text")
        
        arr = json.loads(m.group(0))
        if isinstance(arr, list):
            return [str(x).strip() for x in arr][:3]
        return []
    except Exception as e:
        raise Exception(f"Failed to parse Gemini response: {e}")

def _generate_rules_based_reasons(base: Vehicle, upsell: Vehicle, people: int, luggages: int) -> List[str]:
    """Fallback logic if AI is unavailable."""
    reasons: List[str] = []
    
    def safe_int(x):
        return int(x) if x is not None else None

    base_seats = safe_int(base.seats)
    upsell_seats = safe_int(upsell.seats)
    base_lug = safe_int(base.luggage)
    upsell_lug = safe_int(upsell.luggage)

    reasons.append(f"Spacious interior, provides perfect expierence for 4 people.")
    
    reasons.append(f"Generous trunk space, easily fits all your luggage needs.")
    
    reasons.append(f"Enhanced comfort and premium features for a superior driving experience.")
    
    return reasons[:3]


# ------------------------------
# --- FASTAPI Endpoints ---
# ------------------------------

@app.get("/api/booking/{booking_id}/recommend")
def recommend(booking_id: str, people: Optional[int] = Query(None), luggages: Optional[int] = Query(None)):
    """Fetch vehicles for booking and recommend an upsell using Gemini."""
    
    VEHICLE_API_BASE = "https://hackatum25.sixt.io/"
    url = f"{VEHICLE_API_BASE}/api/booking/{booking_id}/vehicles"
    
    try:
        resp = requests.get(url, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Vehicle fetch failed: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch vehicles from provider")

    # Normalize data structure
    raw_list = []
    if isinstance(data, list):
        raw_list = data
    elif isinstance(data, dict):
        if "deals" in data and isinstance(data["deals"], list):
            raw_list = data["deals"]
        elif "vehicles" in data and isinstance(data["vehicles"], list):
            raw_list = data["vehicles"]
        else:
            raw_list = [data]

    vehicles = [_extract_vehicle_fields(r) for r in raw_list]

    # Select Base Car (lowest price)
    if not vehicles:
         raise HTTPException(status_code=404, detail="No vehicles found")
         
    sorted_by_price = sorted(vehicles, key=lambda v: (v.price if v.price is not None else float("inf")))
    base = sorted_by_price[0]

    # Mock Features if not provided
    if people is None or luggages is None:
        features = get_features_for_booking(booking_id)
        people = people if people is not None else features.get("numberOfPeople", 1)
        luggages = luggages if luggages is not None else features.get("numberOfLuggages", 0)

    # Select Upsell
    other_candidates = [v for v in vehicles if v.id != base.id]
    chosen = choose_best_upsell(base, other_candidates, people, luggages)
    upsell_vehicle = chosen["vehicle"]

    # Generate AI Reasons
    ai_reasons = generate_upsell_reasons(base, upsell_vehicle, people, luggages)

    resp_payload = {
        "bookingId": booking_id,
        "features_used": {"numberOfPeople": people, "numberOfLuggages": luggages},
        "base_car": base.dict(),
        "upsell_car": upsell_vehicle.dict() if upsell_vehicle else None,
        "reason": chosen["reason"],
        "upsell_reasons": ai_reasons,
    }

    # Save to local file for debug/demo
    try:
        with open("car.json", "w", encoding="utf-8") as fh:
            json.dump(resp_payload, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return resp_payload


@app.get("/health/gemini")
def gemini_health():
    """Checks if GEMINI_API_KEY is configured."""
    key = os.environ.get("GEMINI_API_KEY")
    return {
        "gemini_configured": bool(key), 
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    }