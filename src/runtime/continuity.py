from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from runtime.observations import PeriodObservationBatch


@dataclass(frozen=True)
class ContinuityPolicy:
    enabled: bool = True
    max_age_seconds: float = 30.0


@dataclass
class AnalyticsContinuityContext:
    """Small cross-period state needed for temporal correctness."""

    boundary_batch: Optional[PeriodObservationBatch] = None
    counted_ids_by_area: Dict[str, Dict[int, float]] = field(default_factory=dict)
    last_crossing_timestamp_by_area: Dict[str, float] = field(default_factory=dict)


def resolve_continuity_policy(cfg: dict) -> ContinuityPolicy:
    continuity_cfg = cfg.get("continuity", {})
    max_age_seconds = float(continuity_cfg.get("max_age_seconds", 30.0))
    if max_age_seconds < 0:
        raise ValueError("continuity.max_age_seconds must be >= 0")

    return ContinuityPolicy(
        enabled=bool(continuity_cfg.get("enabled", True)),
        max_age_seconds=max_age_seconds,
    )


def boundary_batch_for_next_period(
    batch: PeriodObservationBatch,
    policy: ContinuityPolicy,
) -> Optional[PeriodObservationBatch]:
    if not policy.enabled or batch.frame_count == 0:
        return None

    return batch.boundary_observations(
        end_time=float(batch.period_timestamps[-1]),
        max_age_seconds=policy.max_age_seconds,
    )
