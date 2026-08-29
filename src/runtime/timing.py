from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FrameTiming:
    """Authoritative timing information derived from raw frame metadata."""

    frame_id: int
    timestamp: float
    elapsed_seconds: float
    unix_timestamp: float


class TimingPolicy:
    """Resolve downstream timestamps from raw frame acquisition facts."""

    FRAME = "frame"
    UNIX = "unix"

    def __init__(self, mode: str = FRAME, fps: Optional[float] = None):
        self.mode = self._normalize_mode(mode)
        self.fps = None if fps is None else float(fps)

        if self.mode == self.FRAME and (self.fps is None or self.fps <= 0):
            raise ValueError("timing.mode='frame' requires a positive FPS")

        self._start_frame_id = None
        self._start_timestamp = None

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        fps_override: Optional[float] = None,
    ) -> "TimingPolicy":
        timing_cfg = cfg.get("timing", {})
        mode = timing_cfg.get("mode", cls.FRAME)

        fps = (
            fps_override
            or cfg.get("input", {}).get("fps")
        )

        return cls(mode=mode, fps=fps)

    def resolve(self, frame) -> FrameTiming:
        """Create authoritative timing for a frame-like object."""

        frame_id = int(frame.read_idx)
        unix_timestamp = float(frame.timestamp)

        if self.mode == self.FRAME:
            timestamp = frame_id / self.fps
            if self._start_frame_id is None:
                self._start_frame_id = frame_id
            elapsed_seconds = (frame_id - self._start_frame_id) / self.fps
        else:
            timestamp = unix_timestamp
            if self._start_timestamp is None:
                self._start_timestamp = timestamp
            elapsed_seconds = timestamp - self._start_timestamp

        return FrameTiming(
            frame_id=frame_id,
            timestamp=float(timestamp),
            elapsed_seconds=float(elapsed_seconds),
            unix_timestamp=unix_timestamp,
        )

    def period_elapsed_seconds(
        self,
        period_start: FrameTiming,
        current: FrameTiming,
    ) -> float:
        """Return elapsed period duration including the current frame."""

        if self.mode == self.FRAME:
            return max((current.frame_id - period_start.frame_id + 1) / self.fps, 1.0 / self.fps)

        elapsed = current.timestamp - period_start.timestamp
        if elapsed > 0:
            return float(elapsed)
        if self.fps is not None and self.fps > 0:
            return 1.0 / self.fps
        return 0.0

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = str(mode).lower()
        aliases = {
            "frame": "frame",
            "frame_index": "frame",
            "by_frame": "frame",
            "unix": "unix",
            "wall_clock": "unix",
            "timestamp": "unix",
            "by_timestamp": "unix",
        }
        if normalized not in aliases:
            raise ValueError("timing.mode must be one of: frame, unix")
        return aliases[normalized]
