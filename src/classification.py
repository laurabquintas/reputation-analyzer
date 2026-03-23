"""Shared Ollama-based review classification used by all review scrapers."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

VALID_TOPICS = {"employees", "commodities", "comfort", "cleaning", "quality_price", "meals", "return"}
VALID_SENTIMENTS = {"positive", "negative"}
DEFAULT_MODEL = "qwen2.5:7b"
SUPPORTED_MODELS = {"qwen2.5:7b", "mistral:7b"}

# Preferred detail phrases for constrained vocabulary in prompts
PREFERRED_DETAILS = """\
PREFERRED DETAILS (use these exact phrases when they fit, otherwise write a short custom phrase):
- employees: friendly staff, helpful staff, professional service, attentive staff, rude staff, slow service, unfriendly staff
- commodities: modern hotel, new hotel, great pool, no pool, good wifi, no wifi, good parking, expensive parking, nice facilities, no fitness area, good spa, nice balcony
- comfort: nice room, comfortable bed, spacious room, small room, nice view, quiet location, noisy room, good temperature, poor temperature
- cleaning: very clean, spotless, dirty room, poor hygiene
- quality_price: good value for money, affordable, expensive, overpriced
- meals: good breakfast, varied breakfast, limited breakfast, good dinner, delicious food, repetitive food, expensive food
- return: would return, highly recommend, would not return"""

_SYNONYMS_PATH = Path(__file__).resolve().parent.parent / "data" / "detail_synonyms.json"
_synonyms_cache: dict | None = None


def _load_synonyms() -> dict:
    """Load synonym dict from JSON, cached after first load."""
    global _synonyms_cache
    if _synonyms_cache is None:
        try:
            with open(_SYNONYMS_PATH) as f:
                _synonyms_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _synonyms_cache = {}
    return _synonyms_cache


def _save_synonyms(data: dict) -> None:
    """Write updated synonym dict back to JSON."""
    global _synonyms_cache
    _synonyms_cache = data
    with open(_SYNONYMS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_detail(detail: str, topic: str, sentiment: str) -> str:
    """Normalize a detail phrase via static synonym lookup."""
    detail = detail.strip().lower()
    if not detail:
        return detail
    synonyms = _load_synonyms()
    return synonyms.get(topic, {}).get(sentiment, {}).get(detail, detail)


def batch_normalize_details(
    reviews: list[dict],
    ollama_url: str = "http://localhost:11434",
    model: str = DEFAULT_MODEL,
) -> tuple[int, dict]:
    """Batch-normalize detail phrases using the LLM to find synonym mappings.

    Collects all unique details per topic/sentiment, filters out those already
    in the preferred vocabulary or synonym dict, then asks the LLM to map
    remaining phrases to the closest vocabulary match.

    Returns (num_changed, updated_synonyms_dict).
    """
    # Build the set of preferred (canonical) details per topic, split by sentiment.
    # Positive phrases come before any obviously negative ones in the vocabulary.
    _NEGATIVE_PHRASES = {
        "employees": {"rude staff", "slow service", "unfriendly staff"},
        "commodities": {"no pool", "no wifi", "expensive parking", "no fitness area"},
        "comfort": {"noisy room", "small room", "poor temperature"},
        "cleaning": {"dirty room", "poor hygiene"},
        "quality_price": {"expensive", "overpriced"},
        "meals": {"limited breakfast", "repetitive food", "expensive food"},
        "return": {"would not return"},
    }
    vocab: dict[str, set[str]] = {}
    vocab_by_sentiment: dict[str, dict[str, set[str]]] = {}
    for line in PREFERRED_DETAILS.splitlines():
        if not line.startswith("- "):
            continue
        topic_key, phrases = line[2:].split(":", 1)
        topic_key = topic_key.strip()
        all_phrases = {p.strip().lower() for p in phrases.split(",")}
        vocab[topic_key] = all_phrases
        neg = _NEGATIVE_PHRASES.get(topic_key, set())
        vocab_by_sentiment[topic_key] = {
            "positive": all_phrases - neg,
            "negative": neg,
        }

    synonyms = _load_synonyms()

    # Collect unmapped details per (topic, sentiment)
    unmapped: dict[tuple[str, str], set[str]] = {}
    for review in reviews:
        for t in review.get("topics", []):
            topic = t.get("topic", "")
            sentiment = t.get("sentiment", "")
            detail = t.get("detail", "").strip().lower()
            if not detail or topic not in VALID_TOPICS or sentiment not in VALID_SENTIMENTS:
                continue
            # Skip if already in vocabulary or synonym dict
            if detail in vocab.get(topic, set()):
                continue
            if detail in synonyms.get(topic, {}).get(sentiment, {}):
                continue
            unmapped.setdefault((topic, sentiment), set()).add(detail)

    if not unmapped:
        logger.info("All details already normalized — nothing to do.")
        return 0, synonyms

    # Ask LLM to map each group
    new_mappings: dict[str, dict[str, dict[str, str]]] = {}
    for (topic, sentiment), details in sorted(unmapped.items()):
        # Only offer vocabulary phrases matching the sentiment being normalized
        allowed = sorted(vocab_by_sentiment.get(topic, {}).get(sentiment, set()))
        if not allowed:
            continue
        phrases = sorted(details)
        prompt = (
            f"These are {sentiment} hotel review detail phrases for the topic '{topic}'.\n"
            f"Map each phrase to its CLOSEST SPECIFIC match from ALLOWED.\n"
            f"Only map if the phrase means the SAME specific thing. "
            f"Write KEEP if no ALLOWED phrase captures the same specific meaning.\n"
            f"Do NOT map to a generic phrase — keep the specific detail.\n\n"
            f"ALLOWED: {', '.join(allowed)}\n"
            f"PHRASES: {json.dumps(phrases)}\n\n"
            f"Output ONLY a JSON object mapping each phrase to its match or KEEP. "
            f"No explanation.\n\nJSON:"
        )
        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096},
                },
                timeout=300,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            # Parse JSON from response
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                mapping = json.loads(match.group())
            else:
                mapping = json.loads(raw)
        except Exception as e:
            logger.warning("Failed to normalize %s/%s: %s", topic, sentiment, e)
            continue

        for phrase, target in mapping.items():
            phrase = phrase.strip().lower()
            target = target.strip().lower()
            if target == "keep" or not target:
                continue
            if target in vocab_by_sentiment.get(topic, {}).get(sentiment, set()):
                new_mappings.setdefault(topic, {}).setdefault(sentiment, {})[phrase] = target

    # Merge new mappings into synonyms dict
    for topic, sentiments in new_mappings.items():
        if topic not in synonyms:
            synonyms[topic] = {}
        for sentiment, mappings in sentiments.items():
            if sentiment not in synonyms[topic]:
                synonyms[topic][sentiment] = {}
            synonyms[topic][sentiment].update(mappings)

    _save_synonyms(synonyms)

    # Apply mappings to reviews
    changed = 0
    for review in reviews:
        for t in review.get("topics", []):
            old = t.get("detail", "").strip().lower()
            if not old:
                continue
            new = synonyms.get(t.get("topic", ""), {}).get(t.get("sentiment", ""), {}).get(old)
            if new and new != old:
                t["detail"] = new
                changed += 1

    return changed, synonyms


def is_ollama_available(ollama_url: str = "http://localhost:11434") -> bool:
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def warm_up_model(ollama_url: str = "http://localhost:11434", model: str = DEFAULT_MODEL) -> bool:
    """Pre-load the model into memory with a tiny prompt.

    Call this once before classification to avoid cold-start timeouts.
    """
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 1, "num_ctx": 4096},
            },
            timeout=300,
        )
        return resp.status_code == 200
    except Exception:
        return False


def classify_review(text: str, ollama_url: str = "http://localhost:11434", model: str = DEFAULT_MODEL) -> list[dict]:
    """Classify a review into topics with sentiment using Ollama."""
    prompt = f"""You are a hotel review analyst. Identify ALL topics in this review. The review may be in any language but ALL your output MUST be in English.

TOPICS (use these exact keys):
- employees: staff, service, friendliness, helpfulness, reception, management
- commodities: amenities, facilities, pool, gym, spa, wifi, parking, TV, air conditioning, balcony, shuttle, iron, toiletries, fridge, entertainment, new hotel, modern, renovated
- comfort: room comfort, bed quality, noise, space, temperature, room size, decor, ambiance, view
- cleaning: cleanliness, hygiene, tidiness, housekeeping
- quality_price: value for money, pricing, cost, expensive, affordable
- meals: food, breakfast, restaurant, dining, bar, drinks, buffet, dinner
- return: would return, recommend, come back, revisit

NOTE: wifi, TV, air conditioning, parking = commodities (NOT cleaning or comfort).

{PREFERRED_DETAILS}

RULES:
1. A review CAN have both positive AND negative for the SAME topic.
2. Complaints, "could be better", suggestions for improvement = NEGATIVE.
3. "detail" MUST be a SINGLE phrase in English (translate if needed), 2-4 words, lowercase. Pick ONE phrase from PREFERRED DETAILS when possible. NEVER combine multiple phrases with commas.
4. Output ONLY a JSON array. No explanation, no markdown.

Example: [{{"topic":"employees","sentiment":"positive","detail":"friendly staff"}},{{"topic":"commodities","sentiment":"negative","detail":"no wifi"}}]

Review: \"\"\"{text}\"\"\"

JSON array:"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 768,
            "num_ctx": 4096,
        },
    }

    resp = requests.post(
        f"{ollama_url}/api/generate",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    raw_response = resp.json().get("response", "")
    return _parse_classification(raw_response)


def classify_holidaycheck_review(text: str, ollama_url: str = "http://localhost:11434", model: str = DEFAULT_MODEL) -> list[dict]:
    """Classify a HolidayCheck review using its section labels for better accuracy.

    HolidayCheck reviews contain section labels like ``[Zimmer]``, ``[Service]``,
    ``[Gastronomie]``, etc.  This function tells the LLM what each section means
    so it can assign topics and sentiments more accurately than the generic
    ``classify_review`` prompt.

    Falls back to the generic ``classify_review`` when no section labels are
    detected in the text.
    """
    # If no section labels, fall back to generic classification
    if "[" not in text:
        return classify_review(text, ollama_url, model)

    prompt = f"""You are a hotel review analyst. This German HolidayCheck review has SECTIONS in square brackets. Use them to assign topics. ALL your output MUST be in English.

SECTION → TOPIC: [Zimmer] → comfort/commodities, [Service] → employees, [Gastronomie]/[Restaurant & Bars] → meals, [Lage]/[Lage & Umgebung] → commodities, [Sport & Unterhaltung] → commodities, [Hotel] → commodities/comfort, [Pool]/[Strand] → commodities, [Preis-Leistung] → quality_price, [Allgemein] → any topic.

TOPICS (use these exact keys):
- employees: staff, service, friendliness, helpfulness, reception, management
- commodities: amenities, facilities, pool, gym, spa, wifi, parking, TV, air conditioning, balcony, shuttle, toiletries, entertainment, location, beach, new hotel, modern, renovated
- comfort: room comfort, bed quality, noise, space, temperature, room size, decor, ambiance, view
- cleaning: cleanliness, hygiene, tidiness, housekeeping
- quality_price: value for money, pricing, cost, expensive, affordable
- meals: food, breakfast, restaurant, dining, bar, drinks, buffet, dinner
- return: would return, recommend, come back, revisit

NOTE: wifi, TV, air conditioning, parking = commodities (NOT cleaning or comfort).

{PREFERRED_DETAILS}

RULES:
1. A section CAN have both positive AND negative sentiments.
2. Complaints, suggestions for improvement = NEGATIVE.
3. "detail" MUST be a SINGLE phrase in English (translate if needed), 2-4 words, lowercase. Pick ONE phrase from PREFERRED DETAILS when possible. NEVER combine multiple phrases with commas.
4. Output ONLY a JSON array. No explanation, no markdown.

Example: [{{"topic":"commodities","sentiment":"positive","detail":"modern hotel"}},{{"topic":"commodities","sentiment":"negative","detail":"no wifi"}}]

Review: \"\"\"{text}\"\"\"

JSON array:"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 768,
            "num_ctx": 4096,
        },
    }

    resp = requests.post(
        f"{ollama_url}/api/generate",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    raw_response = resp.json().get("response", "")
    result = _parse_classification(raw_response)

    # Fallback: if the section-aware prompt returned empty, retry with the
    # simpler generic prompt.
    if not result and text.strip():
        logger.info("HolidayCheck prompt returned empty — falling back to generic classify_review.")
        result = classify_review(text, ollama_url, model)

    return result


def classify_booking_review(
    positive_text: str,
    negative_text: str,
    ollama_url: str = "http://localhost:11434",
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    """Classify a Booking.com review using its pre-separated positive/negative text.

    Booking.com reviews already split guest feedback into "Liked" (positive)
    and "Disliked" (negative) sections.  By passing this structure to the LLM
    we eliminate sentiment-guessing errors.
    """
    sections: list[str] = []
    if positive_text:
        sections.append(f"POSITIVE (the guest LIKED this):\n\"\"\"{positive_text}\"\"\"")
    if negative_text:
        sections.append(f"NEGATIVE (the guest DISLIKED this):\n\"\"\"{negative_text}\"\"\"")

    if not sections:
        return []

    review_block = "\n\n".join(sections)

    prompt = f"""You are a hotel review analyst. This Booking.com review is ALREADY separated into POSITIVE (liked) and NEGATIVE (disliked). Use that separation for sentiment. ALL your output MUST be in English.

TOPICS (use these exact keys):
- employees: staff, service, friendliness, helpfulness, reception, management
- commodities: amenities, facilities, pool, gym, spa, wifi, parking, TV, air conditioning, balcony, shuttle, toiletries, entertainment, new hotel, modern, renovated
- comfort: room comfort, bed quality, noise, space, temperature, room size, decor, ambiance, view
- cleaning: cleanliness, hygiene, tidiness, housekeeping
- quality_price: value for money, pricing, cost, expensive, affordable
- meals: food, breakfast, restaurant, dining, bar, drinks, buffet, dinner
- return: would return, recommend, come back, revisit

NOTE: wifi, TV, air conditioning, parking = commodities (NOT cleaning or comfort).

{PREFERRED_DETAILS}

RULES:
1. POSITIVE section → sentiment "positive". NEGATIVE section → sentiment "negative".
2. If NEGATIVE section says NO complaints ("Nothing", "N/A", "Nada", "Nichts", "no complaints"), IGNORE it entirely.
3. "detail" MUST be a SINGLE phrase in English (translate if needed), 2-4 words, lowercase. Pick ONE phrase from PREFERRED DETAILS when possible. NEVER combine multiple phrases with commas.
4. The review may be in any language. Output ONLY a JSON array.

Example: [{{"topic":"meals","sentiment":"positive","detail":"good breakfast"}},{{"topic":"quality_price","sentiment":"negative","detail":"expensive"}}]

{review_block}

JSON array:"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 768,
            "num_ctx": 4096,
        },
    }

    resp = requests.post(
        f"{ollama_url}/api/generate",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    raw_response = resp.json().get("response", "")
    result = _parse_classification(raw_response)

    # Fallback: if the booking-specific prompt returned empty, retry with the
    # simpler generic prompt using the combined text.  This handles short
    # reviews where the structured prompt is too heavy for the model.
    if not result:
        combined = " ".join(filter(None, [positive_text, negative_text])).strip()
        if combined:
            logger.info("Booking prompt returned empty — falling back to generic classify_review.")
            result = classify_review(combined, ollama_url, model)

    return result


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
        # Safety net: if LLM crammed multiple phrases into one detail,
        # keep only the first one.
        if "," in detail:
            detail = detail.split(",")[0].strip()
        pair = (topic, sentiment)
        if topic in VALID_TOPICS and sentiment in VALID_SENTIMENTS and pair not in seen_pairs:
            entry = {"topic": topic, "sentiment": sentiment}
            if detail:
                entry["detail"] = normalize_detail(detail, topic, sentiment)
            result.append(entry)
            seen_pairs.add(pair)

    if not result and items:
        logger.warning("Parsed %d items but none had valid topics: %s", len(items), raw[:200])

    return result
