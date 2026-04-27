# flow_config_normalizer.py
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from pathlib import Path
import yaml


logger = logging.getLogger(__name__)

# allowed counting logic identifiers
_ALLOWED_COUNTING = {"counter_2", "counter_3", "counter_4", "counter_5"}


@dataclass
class FlowConfig:
    enabled: bool
    counting_logic: str
    flow_line_vicinity: float
    time_window_sec: float
    targets: List[str]                      # list of area names
    targets_map: Dict[str, Any]             # name -> area object (for immediate use)


class FlowConfigNormalizer:
    def __init__(self, default_vicinity: float = 0.2, default_time_window: float = 60.0, default_logic: str = "counter_5"):
        self.default_vicinity = float(default_vicinity)
        self.default_time_window = float(default_time_window)
        self.default_logic = default_logic if default_logic in _ALLOWED_COUNTING else "counter_5"

    def _area_has_flow_line(self, area: Any) -> bool:
        """
        Robust check for a flow line in an area object.
        Accepts area.flow_line or area.geometry.get('flow_line') patterns.
        """
        if area is None:
            return False
        # common attribute used by previous loaders
        if hasattr(area, "flow_line") and getattr(area, "flow_line") is not None:
            return True
        # new geometry nesting
        geom = getattr(area, "geometry", None)
        if isinstance(geom, dict):
            fl = geom.get("flow_line") or geom.get("flow_line_points") or geom.get("flow_line_points")
            if fl is not None:
                return True
        # last fallback: attribute named flow_line_points or flow_line_points (string variants)
        if hasattr(area, "flow_line_points") and getattr(area, "flow_line_points") is not None:
            return True
        return False

    def _normalize_targets(self, raw_targets: Any, areas: Sequence[Any]) -> List[str]:
        """Normalize raw targets to list of valid area names that exist in areas list."""
        if raw_targets is None:
            return []

        # produce candidate names
        if isinstance(raw_targets, str):
            candidates = [raw_targets]
        elif isinstance(raw_targets, (list, tuple, set)):
            candidates = [t for t in raw_targets if isinstance(t, str)]
        else:
            logger.info("Flow targets has unsupported type %r — ignoring.", type(raw_targets))
            return []

        # map area names -> objects
        name_to_area = {getattr(a, "name", None): a for a in areas if getattr(a, "name", None) is not None}

        valid: List[str] = []
        for name in candidates:
            area_obj = name_to_area.get(name)
            if area_obj is None:
                logger.info("Flow target '%s' not found among loaded areas — ignoring.", name)
                continue
            # area must be enabled (default True if attribute missing)
            enabled = getattr(area_obj, "enabled", True)
            if enabled is not True:
                logger.info("Flow target '%s' is disabled in areas — ignoring.", name)
                continue
            # area must provide a flow line
            if not self._area_has_flow_line(area_obj):
                logger.info("Flow target '%s' has no valid flow_line -> ignoring.", name)
                continue
            valid.append(name)
        return valid

    def normalize(self, metrics_section: Optional[dict], areas: Sequence[Any]) -> Optional[FlowConfig]:
        """
        metrics_section: dict corresponding to top-level 'metrics' or the 'flow' sub-dict.
                       Prefer passing metrics_section = config.get("metrics", {})
        areas: list of loaded area objects (from AreaConfigLoader.load())

        Returns FlowConfig or None (if flow disabled or no valid targets).
        """
        if not metrics_section:
            logger.info("No metrics section provided -> flow disabled.")
            return None

        flow_raw = metrics_section.get("flow") if "flow" in metrics_section else metrics_section
        if not flow_raw:
            logger.info("No 'flow' config found -> flow disabled.")
            return None

        enabled = bool(flow_raw.get("enabled", True))
        if not enabled:
            logger.info("Flow metric disabled in config.")
            return None

        # normalize targets
        raw_targets = flow_raw.get("targets")
        targets = self._normalize_targets(raw_targets, areas)
        if not targets:
            logger.info("Flow metric: no valid targets after normalization -> disabling flow metric.")
            return None

        # counting logic validation
        logic = str(flow_raw.get("counting_logic") or self.default_logic)
        if logic not in _ALLOWED_COUNTING:
            logger.info("Invalid counting_logic '%s' -> using default '%s'.", logic, self.default_logic)
            logic = self.default_logic

        # flow_line_vicinity
        try:
            vicinity = float(flow_raw.get("flow_line_vicinity", self.default_vicinity))
        except Exception:
            logger.info("Invalid flow_line_vicinity -> using default %.3f", self.default_vicinity)
            vicinity = self.default_vicinity

        # time window
        try:
            time_win = float(flow_raw.get("time_window_sec", self.default_time_window))
            if time_win <= 0:
                raise ValueError("time_window_sec must be > 0")
        except Exception:
            logger.info("Invalid time_window_sec -> using default %.1f", self.default_time_window)
            time_win = self.default_time_window

        # build targets map (name -> area object) for quick access by orchestrator/estimators
        name_to_area = {getattr(a, "name", None): a for a in areas if getattr(a, "name", None) is not None}
        targets_map = {name: name_to_area[name] for name in targets if name in name_to_area}

        return FlowConfig(
            enabled=True,
            counting_logic=logic,
            flow_line_vicinity=vicinity,
            time_window_sec=time_win,
            targets=targets,
            targets_map=targets_map,
        )

    def load_and_normalize(
        self,
        areas: List,
        config_path: Path | str = "configs/traffic_metrics.yaml"
    ) -> Optional[FlowConfig]:
        """
        Load YAML file, extract metrics section,
        and return normalized FlowConfig or None.
        """
        path = Path(config_path)

        if not path.exists():
            logger.error(f"Config file not found: {path}")
            return None

        try:
            with open(path, "r") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception as e:
            logger.error(f"Failed to read config file: {e}")
            return None

        metrics_section = cfg.get("metrics", {})
        return self.normalize(metrics_section, areas)
