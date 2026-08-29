from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml

from geometry.primitives import Area, GeometryEngine, Line, Polygon


SUPPORTED_AREA_TYPES = {"lane", "direction", "mixed", "entire"}
SUPPORTED_COORDINATE_REQUIREMENTS = {"image_or_world", "world_required"}
SUPPORTED_SPATIAL_SCOPES = {"any_area", "lane_only"}


@dataclass(frozen=True)
class MetricCapability:
    name: str
    implemented: bool
    requires_zone: bool
    requires_flow_line: bool


METRIC_CAPABILITIES = {
    "flow": MetricCapability(
        name="flow",
        implemented=True,
        requires_zone=False,
        requires_flow_line=True,
    ),
    "density": MetricCapability(
        name="density",
        implemented=True,
        requires_zone=True,
        requires_flow_line=False,
    ),
    "space_occupancy": MetricCapability(
        name="space_occupancy",
        implemented=False,
        requires_zone=True,
        requires_flow_line=True,
    ),
    "time_occupancy": MetricCapability(
        name="time_occupancy",
        implemented=True,
        requires_zone=True,
        requires_flow_line=True,
    ),
    "space_headway": MetricCapability(
        name="space_headway",
        implemented=True,
        requires_zone=True,
        requires_flow_line=True,
    ),
    "time_headway": MetricCapability(
        name="time_headway",
        implemented=True,
        requires_zone=True,
        requires_flow_line=True,
    ),
    "speed": MetricCapability(
        name="speed",
        implemented=False,
        requires_zone=True,
        requires_flow_line=True,
    ),
}

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_input_config(cfg: dict) -> dict:
    return cfg["input"]


def get_input_source(cfg: dict) -> str:
    source = get_input_config(cfg).get("source")
    if source is None:
        raise ValueError("input.source is required")
    return source


def get_input_fps(cfg: dict) -> float:
    fps = float(get_input_config(cfg)["fps"])
    if fps <= 0:
        raise ValueError("input.fps must be positive")
    return fps


def get_frame_size_reference(cfg: dict) -> int:
    return int(get_geometry_config(cfg)["frame_size_reference"])


def get_polygon_membership_mode(cfg: dict) -> str:
    return get_geometry_config(cfg).get("polygon_membership_mode", "center")


def get_geometry_config(cfg: dict) -> dict:
    return cfg["geometry"]


def get_lines_config(cfg: dict) -> list:
    return get_geometry_config(cfg).get("lines", [])


def get_areas_config(cfg: dict) -> list:
    return get_geometry_config(cfg).get("areas", [])


def get_global_vicinity(cfg: dict) -> Tuple[Optional[float], Optional[float]]:
    vicinity_cfg = get_geometry_config(cfg).get("vicinity", {})
    return vicinity_cfg.get("image"), vicinity_cfg.get("world_m")


def get_metric_config(cfg: dict, metric_name: str) -> dict:
    return cfg.get("metrics", {}).get(metric_name, {})


def coordinate_space_requires_world(cfg: dict) -> bool:
    return str(get_geometry_config(cfg).get("coordinate_space", "image")).lower() == "world"


def has_world_coordinate_capability(cfg: dict) -> bool:
    policy = str(get_geometry_config(cfg).get("coordinate_space", "image")).lower()
    homography_cfg = cfg.get("homography", {})
    homography_enabled = bool(homography_cfg.get("enabled", False))
    calibration_file = homography_cfg.get("calibration_file")
    _, world_vicinity = get_global_vicinity(cfg)

    return (
        policy in {"world", "auto"}
        and homography_enabled
        and bool(calibration_file)
        and world_vicinity is not None
    )


def build_area(
    area_cfg: dict,
    lines_by_id: Dict[str, dict],
    cfg: dict,
) -> Area:
    area_type = area_cfg["area_type"]
    if area_type not in SUPPORTED_AREA_TYPES:
        raise ValueError(f"Unsupported area_type '{area_type}'")

    polygon = None
    if "zone" in area_cfg:
        zone_cfg = area_cfg["zone"]
        polygon = Polygon(
            polygon_id=area_cfg["area_id"],
            points=zone_cfg["points"],
            distance_meters=zone_cfg.get("distance_meters"),
        )

    flow_line = None
    flow_line_id = area_cfg.get("flow_line_id")
    if flow_line_id is not None:
        line_cfg = lines_by_id.get(flow_line_id)
        if line_cfg is None:
            raise ValueError(f"Line '{flow_line_id}' not found")

        vicinity_image, vicinity_world_m = get_global_vicinity(cfg)
        flow_line = Line(
            line_id=line_cfg["line_id"],
            points=line_cfg["points"],
            vicinity=vicinity_image,
            vicinity_world_m=vicinity_world_m,
        )

    return Area(
        area_id=area_cfg["area_id"],
        name=area_cfg.get("name", area_cfg["area_id"]),
        area_type=area_type,
        enable=area_cfg.get("enabled", True),
        description=area_cfg.get("description", ""),
        flow_line=flow_line,
        zone=polygon,
    )


def resolve_eligible_metrics(
    area: Area,
    cfg: dict,
    world_coordinates_available: bool,
) -> Tuple[List[str], Dict[str, str]]:
    eligible = []
    rejected = {}

    for metric_name, metric_cfg in cfg.get("metrics", {}).items():
        capability = METRIC_CAPABILITIES.get(metric_name)
        if capability is None:
            rejected[metric_name] = "unknown metric"
            continue

        if not metric_cfg.get("enabled", False):
            rejected[metric_name] = "disabled"
            continue

        if not capability.implemented:
            rejected[metric_name] = "metric backend is not implemented"
            continue

        spatial_scope = metric_cfg.get("spatial_scope")
        if spatial_scope not in SUPPORTED_SPATIAL_SCOPES:
            rejected[metric_name] = f"unsupported spatial_scope '{spatial_scope}'"
            continue

        if spatial_scope == "lane_only" and area.area_type != "lane":
            rejected[metric_name] = f"spatial_scope '{spatial_scope}' excludes area_type '{area.area_type}'"
            continue

        coordinate_requirement = metric_cfg.get("coordinate_requirement")
        if coordinate_requirement not in SUPPORTED_COORDINATE_REQUIREMENTS:
            rejected[metric_name] = f"unsupported coordinate_requirement '{coordinate_requirement}'"
            continue

        if coordinate_requirement == "world_required" and not world_coordinates_available:
            rejected[metric_name] = "world coordinates unavailable"
            continue

        if capability.requires_zone and area.zone is None:
            rejected[metric_name] = "requires zone"
            continue

        if capability.requires_flow_line and area.flow_line is None:
            rejected[metric_name] = "requires flow line"
            continue

        eligible.append(metric_name)

    return eligible, rejected


def build_areas(
    cfg: dict,
    num_classes: int,
    world_coordinates_available: Optional[bool] = None,
) -> List[Area]:
    if num_classes <= 0:
        raise ValueError("detector num_classes must be positive")

    if world_coordinates_available is None:
        world_coordinates_available = has_world_coordinate_capability(cfg)

    lines_by_id = {
        line_cfg["line_id"]: line_cfg
        for line_cfg in get_lines_config(cfg)
    }
    areas = []

    for area_cfg in get_areas_config(cfg):
        if not area_cfg.get("enabled", True):
            continue

        area = build_area(
            area_cfg=area_cfg,
            lines_by_id=lines_by_id,
            cfg=cfg,
        )
        area.eligible_metrics, area.ineligible_metrics = resolve_eligible_metrics(
            area=area,
            cfg=cfg,
            world_coordinates_available=world_coordinates_available,
        )
        areas.append(area)

    return areas


def build_geometry_engine(
    areas: List[Area],
    cfg: dict,
) -> GeometryEngine:
    lines = {}
    for area in areas:
        if area.flow_line is not None:
            lines[area.flow_line.line_id] = area.flow_line

    polygons = {}
    for area in areas:
        if area.zone is not None:
            polygons[area.zone.polygon_id] = area.zone

    return GeometryEngine(
        lines=lines,
        polygons=polygons,
        frame_size=get_frame_size_reference(cfg),
        polygon_mode=get_polygon_membership_mode(cfg),
        coordinate_space="image",
    )
