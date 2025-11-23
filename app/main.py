import json
import os
import time
from typing import Any, Dict, List, Optional
import socket

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .broadcast_utils import send_broadcast
from .features import extract_features_from_buf
from .addons import fetch_and_save_addons

# Load environment variables
load_dotenv()

app = FastAPI(title="Vehicle Upsell Recommender (OpenAI)")


# --- CLASS DEFINITIONS ---
class Vehicle(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    price: Optional[float] = None
    seats: Optional[int] = None
    luggage: Optional[int] = None
    upsell_titles: List[str] = []
    raw: Optional[Dict[str, Any]] = None
    is_expensive: bool = False


# --- HELPER FUNCTIONS ---


def get_features_for_booking() -> Dict[str, Any]:
    people, luggages, hawaii = extract_features_from_buf()
    people = 3
    luggages = 1
    return {
        "number_of_people": int(people),
        "number_of_luggages": int(luggages),
        "hawaii_shirt_present": hawaii,
    }


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
            elif (
                "displayPrice" in pricing
                and isinstance(pricing["displayPrice"], dict)
                and "amount" in pricing["displayPrice"]
            ):
                price = float(pricing["displayPrice"]["amount"])
    except Exception:
        price = None

    if price is None:
        if (
            isinstance(vehicle_raw, dict)
            and "vehicleCost" in vehicle_raw
            and isinstance(vehicle_raw["vehicleCost"], dict)
            and "value" in vehicle_raw["vehicleCost"]
        ):
            try:
                price = (
                    float(vehicle_raw["vehicleCost"]["value"]) / 100.0
                    if vehicle_raw["vehicleCost"]["value"] > 1000
                    else float(vehicle_raw["vehicleCost"]["value"])
                )
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
        if isinstance(vehicle_raw, dict) and isinstance(
            vehicle_raw.get("attributes"), list
        ):
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

    # Upsell Titles (e.g., "Convertible Luxury")
    upsell_titles = []
    if isinstance(vehicle_raw, dict):
        reasons = vehicle_raw.get("upsellReasons") or []
        for r in reasons:
            if isinstance(r, dict) and "title" in r:
                upsell_titles.append(r["title"])

    is_convertible = False
    
    # Check in name
    if name and "convertible" in name.lower():
        is_convertible = True
    
    # Check in upsell titles
    if not is_convertible:
        for title in upsell_titles:
            if "convertible" in title.lower():
                is_convertible = True
                break

    return Vehicle(
        id=str(vid) if vid is not None else None,
        name=name,
        price=price,
        seats=seats,
        luggage=luggage,
        upsell_titles=upsell_titles,
        raw=deal,
        is_expensive=is_convertible 
    )


# -----------------------------------------------------------
# --- DECISION TREE UPSELL SELECTION ---
# -----------------------------------------------------------


def choose_best_upsell(
    base: Vehicle,
    candidates: List[Vehicle],
    people: int,
    luggages: int,
    hawaii_shirt_present: bool = False,
) -> Dict[str, Any]:
    """
    Selects the best upgrade option using a Decision Tree approach.
    """

    # --- Step 0: Global Filter (Price > 0) ---
    valid_candidates = [v for v in candidates if v.price is not None and v.price > 0.0]

    if not valid_candidates:
        return {"vehicle": None, "reason": "No candidates available with price > 0"}

    base_price = base.price or 0.0

    # --- Step 0.5: Hawaii Override (Convertible Luxury) ---
    if hawaii_shirt_present:
        convertibles = [
            v for v in valid_candidates if "Convertible Luxury" in v.upsell_titles
        ]

        if convertibles:
            # If multiple convertibles exist, pick the cheapest one to increase conversion chance
            best_vehicle = sorted(convertibles, key=lambda x: x.price)[0]
            price_diff = max(0.0, (best_vehicle.price or 0) - base_price)
            return {
                "vehicle": best_vehicle,
                "reason": {
                    "people": people,
                    "luggages": luggages,
                    "decision_path": "hawaii_override_convertible",
                    "description": "Context 'Hawaii' detected. Prioritizing 'Convertible Luxury'.",
                    "price_diff": price_diff,
                },
            }

    # --- Step 1: Define Decision Thresholds ---
    min_seats_required = 5 if people > 2 else people
    min_luggage_required = 4 if luggages >= 1 else luggages

    # --- Step 2: Filter Candidates ---
    strict_matches = []

    for v in valid_candidates:
        v_seats = v.seats if v.seats is not None else 0
        v_luggage = v.luggage if v.luggage is not None else 0

        # Check conditions
        has_enough_seats = v_seats >= min_seats_required
        has_enough_luggage = v_luggage >= min_luggage_required

        if has_enough_seats and has_enough_luggage:
            strict_matches.append(v)

    # --- Step 3: Select Best from Filtered List ---
    best_vehicle = None
    decision_type = ""

    if strict_matches:
        best_vehicle = sorted(strict_matches, key=lambda x: x.price)[0]
        decision_type = "strict_rule_match"
    else:
        # Fallback logic
        seat_matches = [
            v for v in valid_candidates if (v.seats or 0) >= min_seats_required
        ]

        if seat_matches:
            best_vehicle = sorted(seat_matches, key=lambda x: x.price)[0]
            decision_type = "fallback_seats_only"
        else:
            best_vehicle = sorted(valid_candidates, key=lambda x: x.price)[0]
            decision_type = "fallback_cheapest"

    price_diff = max(0.0, (best_vehicle.price or 0) - base_price)

    reason = {
        "people": people,
        "luggages": luggages,
        "min_seats_rule": min_seats_required,
        "min_luggage_rule": min_luggage_required,
        "decision_path": decision_type,
        "price_diff": price_diff,
    }

    return {"vehicle": best_vehicle, "reason": reason}


# -----------------------------------------------------------
# --- GENERATE UPSELL REASONS (OPENAI UPDATED) ---
# -----------------------------------------------------------


def generate_upsell_reasons(
    base: Vehicle,
    upsell: Optional[Vehicle],
    people: int,
    luggages: int,
    hawaii_shirt_present: bool = False,
) -> Dict[str, Any]:
    """
    Generate up to 3 personalized reasons AND a summary sentence using OpenAI.
    """
    if upsell is None:
        return {"reasons": [], "summary": ""}

    # Check for OPENAI key
    openai_key = os.environ.get("OPENAI_API_KEY")
    use_openai = os.environ.get("OPENAI_USE", "1") == "1"

    # 1. Attempt OpenAI Generation
    if openai_key and use_openai:
        try:
            # Pass the Hawaii flag down to the API call
            return _call_openai_api(
                openai_key, base, upsell, people, luggages, hawaii_shirt_present
            )
        except Exception as e:
            print(f"OpenAI Error: {e}")  # Good for debugging
            pass

    # 2. Rule-Based Fallback
    return _generate_rules_based_reasons(base, upsell, people, luggages)


def _call_openai_api(
    api_key: str,
    base: Vehicle,
    upsell: Vehicle,
    people: int,
    luggages: int,
    hawaii_shirt_present: bool,
) -> Dict[str, Any]:
    """
    Internal function to handle the HTTP request to OpenAI (ChatGPT).
    """
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"

    # --- PROMPTS ---

    # Base System Message
    system_msg = (
        "You are a persuasive automotive sales copywriter. You are recommending a premium vehicle upgrade to a customer. "
        "Your goal is to provide convincing arguments why the user should buy the more expensive car based on their specific needs.\n\n"
        "GUIDELINES:\n"
        "1. **NEVER use the word 'upsell'** in the output. Use words like 'upgrade' or 'premium'.\n"
        "2. Focus on specific benefits: interior space for the specific number of passengers, trunk capacity for their luggage, and premium features (speed, comfort, technology).\n"
        "3. Tone: Enthusiastic, professional, and convincing.\n"
        "4. **FORMAT**: Return ONLY a JSON object with exactly two keys:\n"
        "   - 'reasons': An array of 3 concise strings (max 10 words each).\n"
        "   - 'summary': A single, punchy sentence summarizing why they should upgrade.\n"
    )

    # --- HAWAII LOGIC INJECTION ---
    if hawaii_shirt_present:
        system_msg += (
            "\n**CRITICAL CONTEXT**: The customer is wearing a Hawaii shirt. "
            "You must acknowledge they are on a great vacation. "
            "Argue that this specific upgrade (especially if it is a convertible) is the **perfect fit** "
            "to enjoy the sun, the breeze, and the holiday vibes. Make it sound like the ultimate vacation enhancement."
        )
    # ------------------------------

    user_msg = (
        f"Context: The customer is traveling with {people} people and {luggages} pieces of luggage.\n"
        f"Current Base Car: {base.name} (Price: {base.price})\n"
        f"Recommended Upgrade: {upsell.name} (Price: {upsell.price}, Seats: {upsell.seats}, Luggage Capacity: {upsell.luggage})\n\n"
        "Task: Write 3 persuasive bullet points and 1 summary string which uses all 3 arguements from the bullet points. This string should result in exactly 3 sentences. "
        "Output valid JSON."
    )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    body = {
        "model": openai_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "response_format": {"type": "json_object"},
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.7")),
    }

    # API Call with Retries
    max_attempts = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))

    resp = None
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=10.0)
            if resp.status_code == 200:
                break
            if resp.status_code == 429:  # Rate limit
                time.sleep(2**attempt)
                continue
            if 400 <= resp.status_code < 500:  # Client error
                break
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.0)

    if resp is None or resp.status_code != 200:
        error_detail = resp.text if resp else str(last_err)
        raise Exception(f"OpenAI API Error: {error_detail}")

    # Parse Response
    try:
        j = resp.json()
        content = j["choices"][0]["message"]["content"]
        data = json.loads(content)

        reasons = data.get("reasons", [])
        summary = data.get("summary", "")

        if isinstance(reasons, list):
            reasons = [str(x).strip() for x in reasons][:3]
        else:
            reasons = []

        return {"reasons": reasons, "summary": str(summary).strip()}

    except Exception as e:
        raise Exception(f"Failed to parse OpenAI response: {e}")


def _generate_rules_based_reasons(
    base: Vehicle, upsell: Vehicle, people: int, luggages: int
) -> Dict[str, Any]:
    """Fallback logic if AI is unavailable."""
    reasons: List[str] = []

    reasons.append(
        f"Spacious interior, provides perfect experience for {people} people."
    )
    reasons.append(f"Generous trunk space, easily fits all your luggage needs.")
    reasons.append(
        f"Enhanced comfort and premium features for a superior driving experience."
    )

    summary = f"Upgrade to the {upsell.name or 'premium vehicle'} for a more comfortable journey."

    return {"reasons": reasons[:3], "summary": summary}


def generate_additional_driver_text(people: int) -> Dict[str, Any]:
    """
    Returns a short recommendation text (preferably generated by OpenAI) and
    the numeric number of additional driver spots suggested (people - 1, min 0).
    """
    try:
        people = int(people)
    except Exception:
        people = 1

    count = max(0, people - 1)

    openai_key = os.environ.get("OPENAI_API_KEY")
    use_openai = os.environ.get("OPENAI_USE", "1") == "1"

    # Try OpenAI if configured
    if openai_key and use_openai:
        try:
            text = _call_openai_api_for_drivers(openai_key, people, count)
            if isinstance(text, str) and text.strip():
                return {"count": count, "text": text.strip()}
        except Exception as e:
            print(f"OpenAI driver recommendation error: {e}")

    # Fallback rule-based short text (~10 words)
    text = _generate_driver_text_fallback(people, count)
    return {"count": count, "text": text}


def _call_openai_api_for_drivers(api_key: str, people: int, count: int) -> str:
    """
    Calls OpenAI chat completions to generate one short sentence (~10 words)
    recommending the number of Additional Driver spots. Returns plain text.
    """
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"

    system_msg = (
        "You are a concise assistant. Produce exactly one short sentence (~15 words) "
        "that recommends how many Additional Driver spots the customer should buy "
        "based on the number of people. Do not use the word 'upsell'. Keep tone helpful."
    )

    user_msg = (
        f"There are {people} people. Recommend how many Additional Driver spots they should buy. "
        "Return only a single short plain-text sentence, about 15 words. E.g. Currently only one driver can drive. To be more flexible, we recommend adding {count-1} additional driver spots."
    )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": openai_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.7")),
    }

    resp = None
    last_err = None
    max_attempts = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=8.0)
            if resp.status_code == 200:
                break
            if resp.status_code == 429:
                time.sleep(2**attempt)
                continue
            if 400 <= resp.status_code < 500:
                break
        except requests.RequestException as e:
            last_err = e
            time.sleep(0.5)

    if resp is None or resp.status_code != 200:
        error_detail = resp.text if resp else str(last_err)
        raise Exception(f"OpenAI API Error (drivers): {error_detail}")

    try:
        j = resp.json()
        content = j["choices"][0]["message"]["content"]
        return str(content).strip()
    except Exception as e:
        raise Exception(f"Failed to parse OpenAI drivers response: {e}")


def _generate_driver_text_fallback(people: int, count: int) -> str:
    if count <= 0:
        return "Only one driver — no Additional Driver spots needed."
    # Aim for concise ~10-word recommendation
    return f"With {people} people, consider buying {count} Additional Driver spot(s)."


# -----------------------------------------------------------
# --- INSURANCE RECOMMENDATION (NEW) ---
# -----------------------------------------------------------


def generate_insurance_recommendation(is_expensive: bool = False) -> str:
    """
    Generates a specific insurance recommendation text.
    - If is_expensive=True: Recommends premium insurance to protect the high-value vehicle.
    - Else: Recommends insurance based on young driver statistics (accidents doubled under 25).
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    use_openai = os.environ.get("OPENAI_USE", "1") == "1"

    if openai_key and use_openai:
        try:
            text = _call_openai_api_for_insurance(openai_key, is_expensive)
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception as e:
            print(f"OpenAI insurance recommendation error: {e}")

    # Fallback logic if OpenAI fails
    if is_expensive:
        return (
            "For such a premium vehicle, we highly recommend our comprehensive coverage "
            "to ensure your luxury experience remains completely worry-free."
        )
    else:
        return (
            "As a young driver (under 25), statistics show the number of accidents is doubled; "
            "we strongly recommend our comprehensive insurance package for your peace of mind."
        )


def _call_openai_api_for_insurance(api_key: str, is_expensive: bool) -> str:
    """
    Internal call to OpenAI to generate the insurance text.
    Adapts prompt based on whether the car is expensive/convertible.
    """
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"

    if is_expensive:
        system_msg = (
            "You are a luxury mobility consultant. The customer has selected a high-end vehicle. "
            "Write a single, sophisticated sentence recommending Premium Insurance. "
            "Focus on 'protecting this premium experience' and 'peace of mind'. "
            "Do not mention age statistics. Make it sound exclusive and reassuring. "
            "Limit yourself to 10 words."
        )
        user_msg = (
            "The customer chose a convertible/expensive car. Recommend full insurance "
            "to protect this great car and ensure a worry-free trip."
        )
    else:
        system_msg = (
            "You are a helpful and professional insurance advisor at a car rental company. "
            "Your goal is to write a single, persuasive sentence recommending full insurance coverage. "
            "You must explicitly mention that for drivers under 25, the number of accidents is statistically doubled. "
            "Keep it polite but clear about the risk. "
            "Limit yourself to 10 words."
        )
        user_msg = (
            "The customer is under 25 years old. Write a short recommendation (1 sentence) for insurance. "
            "Emphasize that accident rates are double for this age group."
        )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": openai_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.7")),
    }

    resp = None
    last_err = None
    max_attempts = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=8.0)
            if resp.status_code == 200:
                break
            if resp.status_code == 429:
                time.sleep(2**attempt)
                continue
            if 400 <= resp.status_code < 500:
                break
        except requests.RequestException as e:
            last_err = e
            time.sleep(0.5)

    if resp is None or resp.status_code != 200:
        error_detail = resp.text if resp else str(last_err)
        raise Exception(f"OpenAI API Error (insurance): {error_detail}")

    try:
        j = resp.json()
        content = j["choices"][0]["message"]["content"]
        return str(content).strip()
    except Exception as e:
        raise Exception(f"Failed to parse OpenAI insurance response: {e}")


# ------------------------------
# --- FASTAPI Endpoints ---
# ------------------------------


@app.get("/api/booking/{booking_id}/recommend")
def recommend(
    booking_id: str,
    people: Optional[int] = Query(None),
    luggages: Optional[int] = Query(None),
):
    """Fetch vehicles for booking and recommend an upsell using OpenAI."""

    VEHICLE_API_BASE = "https://hackatum25.sixt.io/"
    url = f"{VEHICLE_API_BASE}/api/booking/{booking_id}/vehicles"

    try:
        resp = requests.get(url, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        try:
            with open("all_cars.json", "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except Exception:
            pass
    except Exception as e:
        raise HTTPException(
            status_code=502, detail="Failed to fetch vehicles from provider"
        )

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

    sorted_by_price = sorted(
        vehicles, key=lambda v: (v.price if v.price is not None else float("inf"))
    )
    base = sorted_by_price[0]

    features = get_features_for_booking()
    print(features)

    if people is None:
        people = features.get("number_of_people", 1)
    if luggages is None:
        luggages = features.get("number_of_luggages", 0)

    hawaii_present = features.get("hawaii_shirt_present", False)

    # Select Upsell
    other_candidates = [v for v in vehicles if v.id != base.id]
    chosen = choose_best_upsell(
        base, other_candidates, people, luggages, hawaii_shirt_present=hawaii_present
    )
    upsell_vehicle = chosen["vehicle"]

    # Generate AI Reasons & Summary
    ai_output = generate_upsell_reasons(
        base, upsell_vehicle, people, luggages, hawaii_shirt_present=hawaii_present
    )

    # --- ADDONS SELECTION LOGIC ---
    addons_result = []
    if people >= 2:
        try:
            # Fetch the full JSON data
            data = fetch_and_save_addons(booking_id)

            addons_result = []

            # 1. Check if "addons" key exists and is a list
            if data and "addons" in data and isinstance(data["addons"], list):

                # 2. Iterate through the addon GROUPS (e.g., Child seats, General)
                for group in data["addons"]:

                    # 3. Iterate through the OPTIONS inside each group
                    for option in group.get("options", []):

                        # 4. Check the title inside "chargeDetail"
                        charge_detail = option.get("chargeDetail", {})
                        title = charge_detail.get("title", "")

                        if "additional driver" in title.lower():
                            addons_result = [option]
                            break  # Stop loop once found

                    if addons_result:
                        break  # Stop outer loop if found

            # Output the result
            print(addons_result)

        except Exception as e:
            print(f"An error occurred: {e}")

    # --- Additional Driver Recommendation ---
    additional_driver = generate_additional_driver_text(people)
    additional_driver_count = additional_driver.get("count", 0)
    additional_driver_recommendation = additional_driver.get("text", "")

    # --- NEW: Insurance Recommendation (Young vs Expensive) ---
    # Determine if the upsell car is expensive (convertible)
    is_expensive_car = upsell_vehicle.is_expensive if upsell_vehicle else False
    insurance_text = generate_insurance_recommendation(is_expensive=is_expensive_car)

    resp_payload = {
        "bookingId": booking_id,
        "features_used": {
            "number_of_people": people,
            "number_of_luggages": luggages,
            "hawaii_shirt_present": hawaii_present,
        },
        "base_car": base.dict(),
        "upsell_car": upsell_vehicle.dict() if upsell_vehicle else None,
        "reason": chosen["reason"],
        "upsell_reasons": ai_output.get("reasons", []),
        "upsell_summary": ai_output.get("summary", ""),
        "addons": addons_result,
        "additional_driver_count": additional_driver_count,
        "additional_driver_recommendation": additional_driver_recommendation,
        "insurance_recommendation": insurance_text,
    }

    # Save to local file for debug/demo
    try:
        with open("car.json", "w", encoding="utf-8") as fh:
            json.dump(resp_payload, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return resp_payload


@app.get("/health/ai")
def ai_health():
    """Checks if OPENAI_API_KEY is configured."""
    key = os.environ.get("OPENAI_API_KEY")
    return {
        "openai_configured": bool(key),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    }


# --- UDP Configuration ---
UDP_DESTINATION_PORT = 4210


@app.post("/trigger-broadcast")
def trigger_broadcast():
    """
    Sends a UDP broadcast message 'ready' to the network.
    """
    message = b"ready"
    try:
        send_broadcast(message, UDP_DESTINATION_PORT)
        return {"success": True, "message": "Broadcast sent"}
    except Exception as e:
        print(f"Error sending broadcast: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to send broadcast: {str(e)}"
        )

