from typing import Any, Dict, List, Optional
import os
import logging

import requests
import json
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="Vehicle Upsell Recommender")
logger = logging.getLogger("uvicorn.error")


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
    }

    # Persist the latest recommendation to car.json in the server working directory
    try:
        with open("car.json", "w", encoding="utf-8") as fh:
            json.dump(resp_payload, fh, indent=2, ensure_ascii=False)
        logger.info("Wrote recommendation to car.json")
    except Exception as e:
        logger.exception("Failed to write car.json: %s", e)

    return resp_payload
