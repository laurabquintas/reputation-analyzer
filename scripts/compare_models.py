#!/usr/bin/env python3
"""Compare classification results between two Ollama models side-by-side.

Usage:
    python scripts/compare_models.py                          # uses sample reviews
    python scripts/compare_models.py --source tripadvisor     # uses stored reviews from a source
    python scripts/compare_models.py --source all             # uses stored reviews from all sources
    python scripts/compare_models.py --models qwen2.5:7b mistral:7b
    python scripts/compare_models.py --max-reviews 5          # limit number of reviews

Outputs a side-by-side comparison table and a summary showing which model:
- Extracted more topics
- Used more preferred vocabulary phrases
- Produced fewer hallucinations (details not matching review content)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classification import (
    DEFAULT_MODEL,
    PREFERRED_DETAILS,
    SUPPORTED_MODELS,
    classify_booking_review,
    classify_review,
    is_ollama_available,
    warm_up_model,
)

# ── Sample reviews for quick comparison ──────────────────────────────────────

SAMPLE_REVIEWS = [
    {
        "name": "Melanie (DE)",
        "text": "Wunderschöne Luxus Oase zur perfekten Entspannung. Das Hotel ist wunderschön, sehr modern und liebevoll gestaltet. Sehr schöne Zimmer und ein leckeres Frühstück.",
        "type": "generic",
    },
    {
        "name": "Sonia (EN)",
        "text": "It is a great place to unwind and relax. It is walking distance to the marina and old town. The staff, breakfast and pool area are excellent.",
        "type": "generic",
    },
    {
        "name": "Rita (DE, mixed)",
        "text": "Das Hotel ist ein Traum! Wir hatten ein tolles Zimmer mit Blick auf den Pool. Das Frühstück war abwechslungsreich und lecker. Personal war sehr freundlich. Einziger Nachteil: das WLAN war instabil.",
        "type": "generic",
    },
    {
        "name": "Isabella (IT)",
        "text": "Bellissimo hotel, camere spaziose e pulite. Colazione ottima con grande varietà. Personale gentilissimo. Unico neo: il parcheggio è un po caro. Torneremo sicuramente!",
        "type": "generic",
    },
    {
        "name": "Fátima (PT)",
        "text": "Hotel muito bonito e moderno. Quartos espaçosos e limpos. Pequeno-almoço muito bom com muita variedade. Staff super simpático e prestável. Excelente relação qualidade/preço.",
        "type": "generic",
    },
    {
        "name": "Long DE review",
        "text": "Wir waren ende September im Hotel. Bauarbeiten, die aber nicht sonderlich störten. Die Zimmer groß, geräumig, leider wenig Stauraum. TV sehr intelligent untergebracht, fantastisch!!! Poolbereiche super gepflegt. Es werden Getränke und Snacks am Poolbereich serviert. Sehr ruhiges Hotel. Personal ist sehr freundlich und zuvorkommend. Sprachschwierigkeiten — das Personal spricht oft kein gutes Englisch.",
        "type": "generic",
    },
    {
        "name": "Booking (PT, short)",
        "text": "Muito confortável , hotel novo.",
        "positive_text": "Muito confortável , hotel novo.",
        "negative_text": "Sinceramente não houve nada que não gostasse.",
        "type": "booking",
    },
]


def _build_vocab() -> set[str]:
    """Extract all preferred detail phrases as a set."""
    vocab: set[str] = set()
    for line in PREFERRED_DETAILS.splitlines():
        if not line.startswith("- "):
            continue
        _, phrases = line[2:].split(":", 1)
        for p in phrases.split(","):
            vocab.add(p.strip().lower())
    return vocab


def _classify(review: dict, model: str) -> list[dict]:
    """Classify a review with the given model."""
    if review.get("type") == "booking" and review.get("positive_text"):
        return classify_booking_review(
            review.get("positive_text", ""),
            review.get("negative_text", ""),
            model=model,
        )
    return classify_review(review["text"], model=model)


def _format_topics(topics: list[dict]) -> list[str]:
    """Format topics as short strings."""
    return [
        f"{t['topic']}/{t['sentiment'][:3]} → \"{t.get('detail', '')}\""
        for t in topics
    ]


def _load_source_reviews(source: str, max_reviews: int) -> list[dict]:
    """Load reviews from stored JSON files."""
    if source == "all":
        sources = ["tripadvisor", "booking", "google", "holidaycheck", "expedia"]
    else:
        sources = [source]

    reviews = []
    for s in sources:
        json_path = ROOT / "data" / f"{s}_reviews.json"
        if not json_path.exists():
            print(f"  ⚠ {json_path.name} not found, skipping")
            continue
        with open(json_path) as f:
            data = json.load(f)
        for r in data.get("reviews", []):
            if not r.get("text") and not r.get("positive_text"):
                continue
            entry = {
                "name": f"{s}/{r.get('author_name', 'unknown')[:20]}",
                "text": r.get("text", ""),
                "type": "booking" if s == "booking" else "generic",
            }
            if s == "booking":
                entry["positive_text"] = r.get("positive_text", "")
                entry["negative_text"] = r.get("negative_text", "")
            reviews.append(entry)
            if len(reviews) >= max_reviews:
                return reviews
    return reviews


def main():
    parser = argparse.ArgumentParser(description="Compare Ollama models for review classification")
    parser.add_argument(
        "--models", nargs=2, default=["qwen2.5:7b", "mistral:7b"],
        help="Two models to compare (default: qwen2.5:7b mistral:7b)",
    )
    parser.add_argument(
        "--source", default=None,
        help="Load reviews from stored data: tripadvisor, booking, google, holidaycheck, expedia, or 'all'",
    )
    parser.add_argument(
        "--max-reviews", type=int, default=10,
        help="Max reviews to compare (default: 10)",
    )
    args = parser.parse_args()

    model_a, model_b = args.models

    if not is_ollama_available():
        print("ERROR: Ollama is not running.")
        return 1

    # Load reviews
    if args.source:
        reviews = _load_source_reviews(args.source, args.max_reviews)
        if not reviews:
            print(f"No reviews found for source '{args.source}'")
            return 1
        print(f"Loaded {len(reviews)} reviews from {args.source}")
    else:
        reviews = SAMPLE_REVIEWS[:args.max_reviews]
        print(f"Using {len(reviews)} sample reviews")

    print(f"Comparing: {model_a} vs {model_b}\n")

    # Warm up both models
    print(f"Warming up {model_a}...")
    warm_up_model(model=model_a)
    print(f"Warming up {model_b}...")
    warm_up_model(model=model_b)
    print()

    vocab = _build_vocab()

    # Stats
    stats = {
        model_a: {"total_topics": 0, "vocab_hits": 0, "empty": 0, "time": 0.0},
        model_b: {"total_topics": 0, "vocab_hits": 0, "empty": 0, "time": 0.0},
    }

    separator = "─" * 90
    results_for_report = []

    for i, review in enumerate(reviews, 1):
        name = review["name"]
        text_preview = (review.get("text") or review.get("positive_text", ""))[:80]
        print(f"{separator}")
        print(f"Review {i}/{len(reviews)}: {name}")
        print(f"  \"{text_preview}...\"")
        print()

        # Classify with model A
        t0 = time.time()
        topics_a = _classify(review, model_a)
        time_a = time.time() - t0

        # Classify with model B
        t0 = time.time()
        topics_b = _classify(review, model_b)
        time_b = time.time() - t0

        # Format
        lines_a = _format_topics(topics_a)
        lines_b = _format_topics(topics_b)
        max_lines = max(len(lines_a), len(lines_b), 1)

        # Print side by side
        col_w = 42
        print(f"  {'🅰 ' + model_a:<{col_w}} {'🅱 ' + model_b:<{col_w}}")
        print(f"  {'─' * col_w} {'─' * col_w}")
        for j in range(max_lines):
            left = lines_a[j] if j < len(lines_a) else ""
            right = lines_b[j] if j < len(lines_b) else ""
            print(f"  {left:<{col_w}} {right:<{col_w}}")
        print(f"  {'─' * col_w} {'─' * col_w}")
        print(f"  {len(topics_a)} topics ({time_a:.1f}s){' ' * (col_w - 20)} {len(topics_b)} topics ({time_b:.1f}s)")
        print()

        # Update stats
        for model, topics, t in [(model_a, topics_a, time_a), (model_b, topics_b, time_b)]:
            stats[model]["total_topics"] += len(topics)
            stats[model]["time"] += t
            if not topics:
                stats[model]["empty"] += 1
            for t_entry in topics:
                detail = t_entry.get("detail", "").lower()
                if detail in vocab:
                    stats[model]["vocab_hits"] += 1

        results_for_report.append({
            "name": name,
            "text_preview": text_preview,
            model_a: topics_a,
            model_b: topics_b,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 90}")
    print("SUMMARY")
    print(f"{'═' * 90}\n")

    n = len(reviews)
    header = f"  {'Metric':<35} {model_a:<22} {model_b:<22}"
    print(header)
    print(f"  {'─' * 79}")

    sa, sb = stats[model_a], stats[model_b]

    rows = [
        ("Total topics extracted", str(sa["total_topics"]), str(sb["total_topics"])),
        ("Avg topics per review", f"{sa['total_topics']/n:.1f}", f"{sb['total_topics']/n:.1f}"),
        ("Empty classifications", str(sa["empty"]), str(sb["empty"])),
        ("Preferred vocab usage", f"{sa['vocab_hits']}/{sa['total_topics']}", f"{sb['vocab_hits']}/{sb['total_topics']}"),
        ("Total time", f"{sa['time']:.1f}s", f"{sb['time']:.1f}s"),
        ("Avg time per review", f"{sa['time']/n:.1f}s", f"{sb['time']/n:.1f}s"),
    ]

    for label, va, vb in rows:
        print(f"  {label:<35} {va:<22} {vb:<22}")

    # Vocab percentage
    pct_a = (sa["vocab_hits"] / sa["total_topics"] * 100) if sa["total_topics"] else 0
    pct_b = (sb["vocab_hits"] / sb["total_topics"] * 100) if sb["total_topics"] else 0
    print(f"  {'Vocab hit rate':<35} {pct_a:.0f}%{'':<19} {pct_b:.0f}%")

    print(f"\n{'═' * 90}")

    # Save detailed results as JSON for further analysis
    report_path = ROOT / "data" / "model_comparison.json"
    report = {
        "models": [model_a, model_b],
        "stats": stats,
        "reviews": results_for_report,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
