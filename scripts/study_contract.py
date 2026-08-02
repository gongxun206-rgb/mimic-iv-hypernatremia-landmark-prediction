"""Small data-free helpers mirroring the locked cohort and urine contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def cohort_eligibility(pre_t0_sodium: Iterable[float]) -> dict[str, bool]:
    """Return primary and high-certainty eligibility from pre-T0 sodium values."""
    values = [float(value) for value in pre_t0_sodium]
    all_normal = bool(values) and all(135 <= value <= 145 for value in values)
    return {
        "primary": all_normal,
        "high_certainty": all_normal and len(values) >= 2,
    }


def cleaned_urine_rate(
    candidates: Iterable[Mapping[str, Any]], window_hours: int
) -> tuple[float | None, str | None]:
    """Select the latest valid pre-T0 rate using the locked urine audit rules.

    Each candidate supplies ``hours_before_t0``, ``rate``, ``weight`` and
    ``gu_itemids``. Item IDs 227488 and 227489 mark GU-contaminated windows.
    """
    if window_hours not in (6, 12):
        raise ValueError("Only the locked 6-h and 12-h windows are supported")
    recent = [
        candidate for candidate in candidates
        if 0 < float(candidate["hours_before_t0"]) <= window_hours
        and candidate.get("rate") is not None
    ]
    recent.sort(key=lambda candidate: float(candidate["hours_before_t0"]))
    if not recent:
        return None, "no_recent_measurement"
    for candidate in recent:
        gu_itemids = {int(value) for value in candidate.get("gu_itemids", [])}
        if float(candidate["weight"]) >= 20 and not gu_itemids.intersection({227488, 227489}):
            return float(candidate["rate"]), None
    latest = recent[0]
    latest_gu = {int(value) for value in latest.get("gu_itemids", [])}
    if latest_gu.intersection({227488, 227489}):
        return None, "GU_contaminated"
    if float(latest["weight"]) < 20:
        return None, "implausible_weight"
    return None, "other"


def recorded_outcome(post_t0_sodium: Iterable[float]) -> int | None:
    """Encode recorded Na >=151; no post-T0 measurement remains unclassified."""
    values = [float(value) for value in post_t0_sodium]
    if not values:
        return None
    return int(any(value >= 151 for value in values))
