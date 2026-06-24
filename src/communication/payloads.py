from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def build_metrics_payload(
    sensor_id: str,
    period_result: Dict[str, Any],
    timestamp_utc: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "message_type": "metrics",
        "sensor_id": sensor_id,
        "timestamp_utc": timestamp_utc or utc_now_iso(),
        "period": {
            "period_idx": period_result.get("period_idx"),
            "start_frame": period_result.get("start_frame"),
            "end_frame": period_result.get("end_frame"),
            "frames_processed": period_result.get("frames_processed"),
            "source_frames_elapsed": period_result.get("source_frames_elapsed"),
            "end_of_stream": period_result.get("end_of_stream", False),
        },
        "areas": to_jsonable(period_result.get("area_metrics", {})),
    }


def build_status_payload(
    sensor_id: str,
    online: bool = True,
    processing: bool = False,
    network: bool = True,
    camera: bool = True,
    uptime_sec: Optional[float] = None,
    timestamp_utc: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "message_type": "status",
        "sensor_id": sensor_id,
        "timestamp_utc": timestamp_utc or utc_now_iso(),
        "online": online,
        "processing": processing,
        "network": network,
        "camera": camera,
        "uptime_sec": uptime_sec,
    }
