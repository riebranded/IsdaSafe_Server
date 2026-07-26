from fastapi import HTTPException
from google import genai
from google.genai import types

from app.cache import TTLCache
from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.schemas import FeedingRecommendation, FeedingRecommendationRequest

# Sensor readings only change every 15 minutes, so recommendations for the same
# species/readings are cached for that long instead of calling Gemini each time.
_cache = TTLCache(ttl_seconds=15 * 60)

try:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    _client = genai.Client(api_key=GEMINI_API_KEY)
    _load_error = None
except Exception as exc:  # noqa: BLE001 - report any init failure to the API caller
    _client = None
    _load_error = str(exc)


def _build_prompt(data: FeedingRecommendationRequest) -> str:
    return (
        "You are an aquaculture expert advising a small-scale fish farmer. "
        "Given the species and current water sensor readings below, provide practical, "
        "specific feeding and water-quality guidance.\n\n"
        f"Species: {data.species}\n"
        f"Water temperature: {data.temperature} C\n"
        f"pH: {data.ph}\n"
        f"Dissolved oxygen: {data.dissolved_oxygen} mg/L\n"
        f"Ammonia: {data.ammonia} mg/L\n"
    )


def _request_recommendation(data: FeedingRecommendationRequest) -> FeedingRecommendation:
    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_build_prompt(data),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FeedingRecommendation,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - surface upstream failure without leaking internals
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc

    try:
        return FeedingRecommendation.model_validate_json(response.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"Gemini returned an unparseable response: {exc}"
        ) from exc


def get_feeding_recommendation(data: FeedingRecommendationRequest) -> FeedingRecommendation:
    if _client is None:
        raise HTTPException(
            status_code=503,
            detail=f"Gemini client not configured: {_load_error}",
        )

    return _cache.get_or_set(data.model_dump_json(), lambda: _request_recommendation(data))
