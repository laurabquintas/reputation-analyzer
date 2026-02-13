from bs4 import BeautifulSoup

from src.sites.expedia import (
    _expedia_url_candidates,
    _extract_embedded_json_score,
    _extract_jsonld_score,
    _extract_textual_score,
)


def test_extract_jsonld_score() -> None:
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@context":"https://schema.org","aggregateRating":{"ratingValue":"8.6","bestRating":"10"}}
    </script>
    </head></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_jsonld_score(soup) == 8.6


def test_extract_textual_score() -> None:
    text = "Guest rating 8.6 out of 10 Excellent"
    assert _extract_textual_score(text) == 8.6


def test_extract_embedded_json_score() -> None:
    html = '<script>window.__DATA__={"reviewScoreWithDescription":"8.6 Very good"};</script>'
    assert _extract_embedded_json_score(html) == 8.6


def test_extract_embedded_json_score_with_escaped_rating_value() -> None:
    html = r'<script>window.__STATE__="{\"hotel\":{\"aggregateRating\":{\"ratingValue\":\"8.6\"}}}";</script>'
    assert _extract_embedded_json_score(html) == 8.6


def test_extract_embedded_json_prefers_review_score_over_5_star_classification() -> None:
    html = (
        r'<script>window.__STATE__="{\"property\":{\"classification\":{\"ratingValue\":\"5.0\",\"bestRating\":\"5\"}},'
        r'\"reviews\":{\"aggregateRating\":{\"ratingValue\":\"8.6\",\"bestRating\":\"10\"}}}";</script>'
    )
    assert _extract_embedded_json_score(html) == 8.6


def test_expedia_url_candidates_include_host_and_query_fallbacks() -> None:
    url = (
        "https://euro.expedia.net/Albufeira-Hotels-Example.h123.Hotel-Information"
        "?pwaDialog=product-reviews&foo=bar"
    )
    candidates = _expedia_url_candidates(url)
    assert url in candidates
    assert any("www.expedia.com" in c for c in candidates)
    assert any("www.expedia.co.uk" in c for c in candidates)
    assert any("foo=bar" in c and "pwaDialog" not in c for c in candidates)
