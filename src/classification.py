"""Shared Ollama-based review classification used by all review scrapers."""

from __future__ import annotations

import json
import logging
import re

import requests

logger = logging.getLogger(__name__)

VALID_TOPICS = {"employees", "commodities", "comfort", "cleaning", "quality_price", "meals", "return"}
VALID_SENTIMENTS = {"positive", "negative"}


def is_ollama_available(ollama_url: str = "http://localhost:11434") -> bool:
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def classify_review(text: str, ollama_url: str = "http://localhost:11434") -> list[dict]:
    """Classify a review into topics with sentiment using Ollama."""
    prompt = f"""You are a hotel review analyst. Read the review below carefully and identify ALL topics mentioned, even briefly. Pay special attention to complaints, cons, criticisms, and suggestions for improvement – these are NEGATIVE.

TOPICS (use these exact keys):
- employees: any mention of staff, service, friendliness, helpfulness, reception, concierge, team, waiters, management
- commodities: amenities, facilities, pool, gym, spa, room features, wifi, parking, fridge, toiletries, TV, air conditioning, balcony, shuttle, iron, entertainment, music
- comfort: room comfort, bed quality, noise, quiet, space, temperature, room size, mattress, pillow, decor, ambiance, construction noise, view
- cleaning: cleanliness, hygiene, tidiness, housekeeping, spotless, dirty, stains, towels changed, room serviced
- quality_price: value for money, pricing, worth, cost, overpriced, good deal, expensive, cheap, affordable, half board value
- meals: food, breakfast, restaurant, dining, bar, drinks, buffet, dinner, lunch, cuisine, menu, chef, kitchen, snacks, repetitive food, variety
- return: whether the guest would return, come back, visit again, recommend, revisit, not return, wouldn't go back

RULES:
1. You MUST check each topic one by one. Go through the review sentence by sentence.
2. A single review CAN and OFTEN DOES have both positive AND negative for the SAME topic. For example breakfast can be praised (positive) but also called repetitive (negative).
3. Even brief or indirect mentions count (e.g. "rooms were cleaned daily" = cleaning positive).
4. If a topic is described positively, mark it positive. If negatively, mark it negative.
5. Complaints, cons, "could be better", "didn't work well", "wish they had", suggestions for improvement = NEGATIVE. Do NOT skip these.
6. For each entry include a "detail" field: a short phrase (2-4 words) describing the specific aspect mentioned. Use lowercase.
7. Output ONLY a JSON array. No explanation, no markdown.

EXAMPLE INPUT: "Staff were amazing. Breakfast was varied but got repetitive after a few days. Pool was cold and could do with music but rooms were spotless and spacious. The air con struggled to keep the room cool. Would definitely come back!"
EXAMPLE OUTPUT: [{{"topic":"employees","sentiment":"positive","detail":"amazing staff"}},{{"topic":"meals","sentiment":"positive","detail":"varied breakfast"}},{{"topic":"meals","sentiment":"negative","detail":"repetitive breakfast"}},{{"topic":"commodities","sentiment":"negative","detail":"cold pool"}},{{"topic":"cleaning","sentiment":"positive","detail":"spotless rooms"}},{{"topic":"comfort","sentiment":"positive","detail":"spacious rooms"}},{{"topic":"comfort","sentiment":"negative","detail":"poor air conditioning"}},{{"topic":"return","sentiment":"positive","detail":"would come back"}}]

Now analyze this review:
\"\"\"{text}\"\"\"

JSON array:"""

    payload = {
        "model": "mistral:7b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 768,
        },
    }

    resp = requests.post(
        f"{ollama_url}/api/generate",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    raw_response = resp.json().get("response", "")
    return _parse_classification(raw_response)


def _parse_classification(raw: str) -> list[dict]:
    """Parse Ollama JSON response into validated topic classifications."""
    cleaned = raw.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    items = None

    # Try 1: parse the whole string as JSON
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try 2: extract a JSON array from the string
    if items is None:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Try 3: extract individual objects via regex
    # This handles malformed arrays like [{"topic":"x","sentiment":"y"},{"topic":"z":null}]
    if items is None:
        pair_matches = re.findall(
            r'\{\s*"topic"\s*:\s*"([^"]+)"\s*,\s*"sentiment"\s*:\s*"([^"]+)"'
            r'(?:\s*,\s*"detail"\s*:\s*"([^"]*)")?\s*\}',
            cleaned,
        )
        if pair_matches:
            items = [{"topic": t, "sentiment": s, "detail": d} for t, s, d in pair_matches]

    if items is None or not isinstance(items, list):
        if cleaned:
            logger.warning("Failed to parse Ollama response: %s", raw[:200])
        return []

    # Allow same topic with both positive AND negative (but not duplicate pairs)
    seen_pairs: set[tuple[str, str]] = set()
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic", "")).lower().strip()
        sentiment = str(item.get("sentiment", "")).lower().strip()
        detail = str(item.get("detail", "")).strip()
        pair = (topic, sentiment)
        if topic in VALID_TOPICS and sentiment in VALID_SENTIMENTS and pair not in seen_pairs:
            entry = {"topic": topic, "sentiment": sentiment}
            if detail:
                entry["detail"] = detail
            result.append(entry)
            seen_pairs.add(pair)

    return result
