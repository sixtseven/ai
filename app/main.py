from typing import Any, Dict, List, Optional
import os
import logging

import requests
import json
import time
import random
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Vehicle Upsell Recommender")
# Configure logging to write to out.txt instead of printing to console.
# Use basicConfig with force=True to override existing handlers (available on Python 3.8+).
LOG_FILE = os.environ.get("LOG_FILE", "out.txt")
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

# Ensure uvicorn's logger does not re-add console handlers; get a named logger for use in the app.
logger = logging.getLogger("uvicorn.error")
logger.propagate = False


class Vehicle(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    price: Optional[float] = None
    seats: Optional[int] = None
    luggage: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None


def get_features_for_booking(booking_id: str) -> Dict[str, int]:
    """Mocked features provider. Replace with your real service later.

    Returns a dict containing numberOfPeople and numberOfLuggages.
    """
    # Simple deterministic mock based on booking id to make testing repeatable
    try:
        v = abs(hash(booking_id))
    except Exception:
        v = 42
    people = 2 + (v % 4)  # 2..5
    luggages = 0 + (v % 3)  # 0..2
    return {"numberOfPeople": int(people), "numberOfLuggages": int(luggages)}


def _extract_vehicle_fields(raw: Dict[str, Any]) -> Vehicle:
    # The payload might be a deal object with nested 'vehicle' and 'pricing' (Sixt hackatum style)
    deal = raw
    vehicle_raw = raw
    pricing = None
    # If this is a deal wrapper, unwrap
    if isinstance(raw, dict) and "vehicle" in raw:
        vehicle_raw = raw.get("vehicle") or {}
        pricing = raw.get("pricing") or {}

    # ids/names
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

    # price: prefer pricing.totalPrice.amount, then displayPrice.amount, else top-level price keys
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
        # fallback to vehicleCost or common keys
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

    # seats and luggage
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

    luggage = None
    if isinstance(vehicle_raw, dict):
        luggage = vehicle_raw.get("bagsCount") or vehicle_raw.get("luggage")
        if luggage is not None:
            try:
                luggage = int(luggage)
            except Exception:
                luggage = None

    if luggage is None:
        # try attributes inside vehicle_raw
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
    """Choose best upsell candidate given features.

    Strategy:
    - Prefer vehicles that satisfy seats >= people and luggage >= luggages
    - Among those, choose minimal price increase over base
    - If none satisfy, pick vehicle minimizing total shortage (people_shortage + luggage_shortage) and then minimal price increase
    """
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

        # score tuple: prefer zero shortages, then minimal price_diff, then more seats
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


def generate_upsell_reasons(base: Vehicle, upsell: Optional[Vehicle], people: int, luggages: int) -> List[str]:
    """Generate up to 3 personalized, human-readable reasons to upsell.

    Reasons prioritize capacity (seats), luggage space, and useful features/value.
    """
    if upsell is None:
        return []

    # If OPENAI_API_KEY is set and OPENAI_USE is truthy, prefer generating polished upsell points via ChatGPT
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_use_flag = os.environ.get("OPENAI_USE", "1")
    use_chat = bool(openai_key) and openai_use_flag not in ("0", "false", "False", "no", "NONE", "")
    if use_chat:
        try:
            # prepare a compact payload with necessary fields
            payload = {
                "base_car": {
                    "name": base.name,
                    "seats": base.seats,
                    "luggage": base.luggage,
                    "price": base.price,
                },
                "upsell_car": {
                    "name": upsell.name,
                    "seats": upsell.seats,
                    "luggage": upsell.luggage,
                    "price": upsell.price,
                },
                "features": {"people": people, "luggages": luggages}
            }

            system_msg = (
                "You are an expert marketing copywriter. Given a customer's current booked car and a recommended upgrade, "
                "produce exactly three concise, persuasive, and honest upsell bullets that would motivate the customer to choose the upgrade. "
                "Each bullet should be one sentence, positive, non-judgmental, and relevant to the customer's party size and luggage. "
                "Return ONLY a JSON array of three strings, no extra text."
            )

            user_msg = (
                f"Base car: {json.dumps(payload['base_car'])}\n"
                f"Upsell car: {json.dumps(payload['upsell_car'])}\n"
                f"Customer: {people} people, {luggages} luggage items.\n\n"
                "Provide the three upsell bullets as a JSON array."
            )

            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.8,
                "max_tokens": 300,
            }

            logger.info("Calling OpenAI Chat Completions model %s", body.get("model"))
            # Retry logic for transient errors (429 rate limit, 5xx)
            max_attempts = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))
            backoff_base = float(os.environ.get("OPENAI_BACKOFF_BASE", "1.0"))
            resp = None
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={**headers, "Accept": "application/json"},
                        json=body,
                        timeout=15.0,
                    )
                    logger.info("OpenAI responded with status %s on attempt %d", resp.status_code, attempt)
                    # If success or client error other than 429, break
                    if resp.status_code == 200 or (400 <= resp.status_code < 500 and resp.status_code != 429):
                        break
                    # If rate limited (429) or server error (5xx), retry
                except requests.RequestException as e:
                    logger.warning("OpenAI request attempt %d failed: %s", attempt, e)
                    resp = None

                # Exponential backoff with jitter before next attempt
                if attempt < max_attempts:
                    sleep_time = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.info("Retrying OpenAI request in %.2fs (attempt %d/%d)", sleep_time, attempt + 1, max_attempts)
                    time.sleep(sleep_time)
        except requests.RequestException as e:
            # Hard fail when OpenAI is configured but cannot be reached
            logger.exception("OpenAI request failed")
            raise HTTPException(status_code=502, detail=f"Failed to call OpenAI API: {e}")

        # If OpenAI returned an error status, decide how to proceed
        if resp is None:
            logger.error("OpenAI request failed after retries; no response object")
            raise HTTPException(status_code=502, detail="OpenAI API request failed after retries")

        if resp.status_code == 429:
            # Rate limit — inform the caller and include response body for debugging
            logger.error("OpenAI rate limited: %s", resp.text)
            raise HTTPException(status_code=429, detail=f"OpenAI rate limit: {resp.text}")

        if resp.status_code != 200:
            logger.error("OpenAI API returned status %s: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=502, detail=f"OpenAI API error: {resp.status_code}")

        try:
            j = resp.json()
            # Extract assistant content
            content = None
            try:
                content = j["choices"][0]["message"]["content"]
            except Exception:
                content = j["choices"][0].get("text") if j.get("choices") else None

            if not content:
                raise HTTPException(status_code=502, detail="OpenAI returned empty completion")

            # The assistant should return a JSON array. Try to parse first JSON block.
            text = content.strip()
            import re
            m = re.search(r"\[.*\]", text, flags=re.S)
            if not m:
                raise HTTPException(status_code=502, detail="OpenAI response did not contain a JSON array")
            arr_text = m.group(0)
            arr = json.loads(arr_text)
            if not isinstance(arr, list):
                raise HTTPException(status_code=502, detail="OpenAI response JSON was not an array")
            # ensure strings
            return [str(x).strip() for x in arr][:3]
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to parse OpenAI response")
            raise HTTPException(status_code=502, detail=f"Failed to parse OpenAI response: {e}")

    reasons: List[str] = []

    # Helpers
    def safe_int(x):
        try:
            return int(x) if x is not None else None
        except Exception:
            return None

    base_seats = safe_int(base.seats)
    upsell_seats = safe_int(upsell.seats)
    base_lug = safe_int(base.luggage)
    upsell_lug = safe_int(upsell.luggage)

    # 1) Capacity reason (people)
    if upsell_seats is not None and (base_seats is None or upsell_seats >= people and (base_seats < people if base_seats is not None else True)):
        reasons.append(f"Fits your party: the upsell offers {upsell_seats} seats vs {base_seats or 'fewer'}, for your group of {people} people.")

    # 2) Luggage reason
    if upsell_lug is not None and (base_lug is None or upsell_lug >= luggages and (base_lug < luggages if base_lug is not None else True)):
        reasons.append(f"More trunk space: the upsell fits {upsell_lug} bags vs {base_lug or 'less'}, covering your {luggages} luggage items.")

    # 3) Feature/value reason
    # Inspect upsell.raw for common upsell attributes
    feature_reasons: List[str] = []
    try:
        raw = upsell.raw or {}
        vehicle = raw.get("vehicle") if isinstance(raw, dict) else None
        # prefer flags
        if vehicle and isinstance(vehicle, dict):
            if vehicle.get("isRecommended"):
                feature_reasons.append("Recommended vehicle — our system flags this as a good option.")
            # attributes list
            attrs = vehicle.get("attributes") or []
            for a in attrs:
                title = (a.get("title") or "").lower()
                val = a.get("value") or ""
                if "navigation" in title or "built-in navigation" in title:
                    feature_reasons.append("Includes built-in navigation for easier driving in unfamiliar areas.")
                if "new vehicle" in title or vehicle.get("isNewCar"):
                    feature_reasons.append("Newer car model — more comfort and reliability.")
                if "electric" in (vehicle.get("fuelType") or "").lower():
                    feature_reasons.append("Electric vehicle — quieter ride and lower local emissions.")
                if "boot" in title or "trunk" in title or "bags" in title:
                    # already handled by luggage, but can mention
                    feature_reasons.append(f"Trunk capacity: {val}.")
    except Exception:
        feature_reasons = []

    # Add the top unique feature reasons
    for fr in feature_reasons:
        if len(reasons) >= 3:
            break
        if fr not in reasons:
            reasons.append(fr)

    # If still fewer than 3, add a price/value reason
    if len(reasons) < 3:
        try:
            base_price = float(base.price) if base.price is not None else 0.0
            upsell_price = float(upsell.price) if upsell.price is not None else 0.0
            price_diff = upsell_price - base_price
            if price_diff <= 0:
                reasons.append("No additional daily cost for this upgrade — great value.")
            else:
                reasons.append(f"Small price increase of {price_diff:.2f} (total) for more space/features.")
        except Exception:
            reasons.append("Upgrade offers more space and comfort compared to your current car.")

    # Ensure at most 3 reasons
    return reasons[:3]


@app.get("/api/booking/{booking_id}/recommend")
def recommend(booking_id: str, people: Optional[int] = Query(None), luggages: Optional[int] = Query(None)):
    """Fetch vehicles for booking and recommend an upsell along with the base car.

    Query params 'people' and 'luggages' can be used to override the mocked feature provider for testing.
    """
    VEHICLE_API_BASE = "https://hackatum25.sixt.io/"

    # Fetch vehicles from external endpoint
    url = f"{VEHICLE_API_BASE}/api/booking/{booking_id}/vehicles"
    try:
        resp = requests.get(url, timeout=5.0)
    except requests.RequestException as e:
        logger.exception("Failed to fetch vehicles")
        raise HTTPException(status_code=502, detail=f"Failed to fetch vehicles: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Vehicle service returned {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Vehicle service returned non-JSON response")

    # data might be list or dict
    raw_list = []
    if isinstance(data, list):
        raw_list = data
    elif isinstance(data, dict):
        # common key 'deals' (Sixt hackatum) or 'vehicles'
        if "deals" in data and isinstance(data["deals"], list):
            raw_list = data["deals"]
        elif "vehicles" in data and isinstance(data["vehicles"], list):
            raw_list = data["vehicles"]
        else:
            # fallback: treat dict as single vehicle/deal
            raw_list = [data]
    else:
        raise HTTPException(status_code=502, detail="Unexpected vehicle service payload format")

    vehicles = [_extract_vehicle_fields(r) for r in raw_list]

    # determine base car: prefer vehicle with price == 0, else minimum price
    base_candidates = [v for v in vehicles if (v.price is not None and v.price == 0.0)]
    if base_candidates:
        base = base_candidates[0]
    else:
        # choose min price
        sorted_by_price = sorted(vehicles, key=lambda v: (v.price if v.price is not None else float("inf")))
        base = sorted_by_price[0] if sorted_by_price else None

    if base is None:
        raise HTTPException(status_code=502, detail="No vehicles returned by vehicle service")

    # features (mocked) unless overridden by query params
    if people is None or luggages is None:
        features = get_features_for_booking(booking_id)
        people = people if people is not None else features.get("numberOfPeople", 1)
        luggages = luggages if luggages is not None else features.get("numberOfLuggages", 0)

    # candidates excluding base
    other_candidates = [v for v in vehicles if v is not base]

    chosen = choose_best_upsell(base, other_candidates, people, luggages)

    resp_payload = {
        "bookingId": booking_id,
        "features_used": {"numberOfPeople": people, "numberOfLuggages": luggages},
        "base_car": base.dict(),
        "upsell_car": chosen["vehicle"].dict() if chosen["vehicle"] is not None else None,
        "reason": chosen["reason"],
        "upsell_reasons": generate_upsell_reasons(base, chosen["vehicle"], people, luggages),
    }

    # Persist the latest recommendation to car.json in the server working directory
    try:
        with open("car.json", "w", encoding="utf-8") as fh:
            json.dump(resp_payload, fh, indent=2, ensure_ascii=False)
        logger.info("Wrote recommendation to car.json")
    except Exception as e:
        logger.exception("Failed to write car.json: %s", e)

    return resp_payload


@app.get("/health/openai")
def openai_health():
    """Simple health endpoint to check if OPENAI_API_KEY is configured and reachable.

    Returns 200 if OPENAI_API_KEY is set. Does NOT call OpenAI. Use this to confirm server-side config.
    """
    key = os.environ.get("OPENAI_API_KEY")
    enabled = bool(key)
    return {"openai_configured": enabled, "openai_use_flag": os.environ.get("OPENAI_USE", "1")}
