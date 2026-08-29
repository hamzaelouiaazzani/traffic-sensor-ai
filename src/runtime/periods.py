from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


MAX_PERIOD_SECONDS = 3600.0
DEFAULT_MAX_OBS_PER_FRAME = 50
MIN_MAX_OBS_PER_FRAME = 10
MAX_MAX_OBS_PER_FRAME = 50


@dataclass(frozen=True)
class PeriodPolicy:
    period_seconds: float
    fps: float
    max_obs_per_frame: int
    max_observations: int


def resolve_period_policy(
    cfg: dict,
    fps: float,
    period_mins_override: Optional[float] = None,
) -> PeriodPolicy:
    period_seconds = resolve_period_seconds(
        cfg=cfg,
        period_mins_override=period_mins_override,
    )
    max_obs_per_frame = resolve_max_obs_per_frame(cfg)
    max_observations = derive_max_observations(
        period_seconds=period_seconds,
        fps=fps,
        max_obs_per_frame=max_obs_per_frame,
    )
    return PeriodPolicy(
        period_seconds=period_seconds,
        fps=float(fps),
        max_obs_per_frame=max_obs_per_frame,
        max_observations=max_observations,
    )


def resolve_period_seconds(
    cfg: dict,
    period_mins_override: Optional[float] = None,
) -> float:
    if period_mins_override is not None:
        period_seconds = float(period_mins_override) * 60.0
    else:
        period_seconds = float(cfg["analytics"]["period_seconds"])

    if period_seconds <= 0:
        raise ValueError("analytics.period_seconds must be positive")
    return min(period_seconds, MAX_PERIOD_SECONDS)


def resolve_max_obs_per_frame(cfg: dict) -> int:
    value = int(cfg.get("analytics", {}).get("max_obs_per_frame", DEFAULT_MAX_OBS_PER_FRAME))
    return min(max(value, MIN_MAX_OBS_PER_FRAME), MAX_MAX_OBS_PER_FRAME)


def derive_max_observations(
    period_seconds: float,
    fps: float,
    max_obs_per_frame: int,
) -> int:
    fps = float(fps)
    if fps <= 0:
        raise ValueError("FPS must be positive")

    period_seconds = float(period_seconds)
    if period_seconds <= 0:
        raise ValueError("period_seconds must be positive")

    return max(1, int(period_seconds * fps * int(max_obs_per_frame)))
