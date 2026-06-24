import yaml

from geometry.primitives import Line, Polygon, Area, GeometryEngine
from traffic_metrics.metrics import Counter, FlowMetric, DensityMetric, OccupancyMetric, TimeHeadwayMetric, SpaceHeadwayMetric
from crossing.crossing_estimation import CrossingExtractor


METRIC_REGISTRY = {

    "counter": {
        "class": Counter,

        "params": {
            "counter_logic": lambda cfg, area:
                cfg["metrics"]["flow"]["counter_logic"],

            "ttl_seconds": lambda cfg, area:
                cfg["metrics"]["flow"]["ttl_seconds"],

            "num_classes": lambda cfg, area:
                cfg["general_params"]["num_classes"],
        },
    },

    "flow": {
        "class": FlowMetric,

        "params": {

            "num_classes": lambda cfg, area:
                cfg["general_params"]["num_classes"],
        },
    },

    "density": {
        "class": DensityMetric,
    
        "params": {
            "distance_meters": lambda cfg, area:
                area.zone.distance_meters,
        },
    },
    
    "space_headway": {
        "class": SpaceHeadwayMetric,
    
        "params": {
            "direction": lambda cfg, area:
                (area.flow_line._A, area.flow_line._B),
        },
    },


"occupancy": {
    "class": OccupancyMetric,

    "params": {},
},



"time_headway": {
    "class": TimeHeadwayMetric,

    "params": {},
},
    
}





def load_config(path: str):

    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    return raw_cfg





def build_area(
    area_cfg: dict,
    lines_cfg: list,
) -> Area:

    # =================================================
    # Polygon
    # =================================================

    polygon = None

    if "zone" in area_cfg:

        zone_cfg = area_cfg["zone"]

        polygon = Polygon(
            polygon_id=area_cfg["area_id"],
            points=zone_cfg["points"],
            distance_meters=zone_cfg.get(
                "distance_meters"
            ),
        )

    # =================================================
    # Flow line
    # =================================================

    flow_line = None

    flow_line_id = area_cfg.get("flow_line_id")

    if flow_line_id is not None:

        line_cfg = next(
            (
                l for l in lines_cfg
                if l["line_id"] == flow_line_id
            ),
            None
        )

        if line_cfg is None:
            raise ValueError(
                f"Line '{flow_line_id}' not found"
            )

        flow_line = Line(
            line_id=line_cfg["line_id"],
            points=line_cfg["points"],
            vicinity=line_cfg.get("vicinity"),
        )

    # =================================================
    # Area
    # =================================================

    area = Area(
        area_id=area_cfg["area_id"],
        name=area_cfg["name"],
        enable=area_cfg.get("enable", True),
        description=area_cfg.get(
            "description",
            ""
        ),
        flow_line=flow_line,
        zone=polygon,
        metrics_names=area_cfg.get(
            "metrics",
            []
        ),
    )

    return area



def get_eligible_metrics(
    metrics: list[str],
    zone_flag: bool,
    flow_line_flag: bool,
) -> list[str]:

    eligible = []

    for metric in metrics:

        # flow -> needs flow line
        if metric == "flow":

            if flow_line_flag:
                eligible.append(metric)

        # density -> needs only zone
        elif metric == "density":

            if zone_flag:
                eligible.append(metric)

        # others -> need both
        elif metric in {
            "occupancy",
            "space_headway",
            "time_headway",
        }:

            if zone_flag and flow_line_flag:
                eligible.append(metric)

    return eligible


    
def assign_eligible_metrics(
    area: Area,
    metrics: list[str],
) -> None:

    zone_flag = area.zone is not None

    flow_line_flag = area.flow_line is not None

    area.eligible_metrics = get_eligible_metrics(
        metrics=metrics,
        zone_flag=zone_flag,
        flow_line_flag=flow_line_flag,
    )



def build_metric_objects(
    area: Area,
    cfg: dict,
) -> Area:

    # ---------------------------------
    # Empty metrics
    # ---------------------------------

    area.metrics = {}

    if not area.eligible_metrics:
        return area

    # ---------------------------------
    # Build metrics
    # ---------------------------------

    for metric_name in area.eligible_metrics:

        metric_spec = METRIC_REGISTRY[
            metric_name
        ]

        metric_cls = metric_spec["class"]

        param_builders = metric_spec[
            "params"
        ]

        # -------------------------
        # Resolve params
        # -------------------------

        params = {
            param_name: resolver(cfg, area)
            for param_name, resolver
            in param_builders.items()
        }

        # -------------------------
        # Instantiate metric
        # -------------------------

        area.metrics[metric_name] = (
            metric_cls(**params)
        )

    # ---------------------------------
    # Counter special case
    # ---------------------------------

    if "flow" in area.eligible_metrics:

        counter_spec = METRIC_REGISTRY[
            "counter"
        ]

        counter_cls = counter_spec["class"]

        counter_params = {
            param_name: resolver(cfg, area)
            for param_name, resolver
            in counter_spec["params"].items()
        }

        area.metrics["counter"] = (
            counter_cls(**counter_params)
        )

    return area




def build_areas(
    cfg: dict,
) -> list[Area]:

    areas = []

    areas_cfg = cfg["areas"]
    lines_cfg = cfg["lines"]

    for area_cfg in areas_cfg:

        # ---------------------------------
        # Build area
        # ---------------------------------

        area = build_area(
            area_cfg=area_cfg,
            lines_cfg=lines_cfg,
        )

        # ---------------------------------
        # Eligible metrics
        # ---------------------------------

        assign_eligible_metrics(
            area=area,
            metrics=area_cfg.get(
                "metrics",
                []
            ),
        )

        # ---------------------------------
        # Instantiate metric objects
        # ---------------------------------

        area = build_metric_objects(
            area=area,
            cfg=cfg,
        )

        areas.append(area)

    return areas



def build_runtime_components(
    areas: list[Area],
    cfg: dict,
):

    # =================================================
    # Extract unique lines
    # =================================================

    lines = {}

    for area in areas:

        if area.flow_line is not None:

            lines[
                area.flow_line.line_id
            ] = area.flow_line

    # =================================================
    # Extract polygons
    # =================================================

    polygons = {}

    for area in areas:

        if area.zone is not None:

            polygons[
                area.zone.polygon_id
            ] = area.zone

    # =================================================
    # Geometry Engine
    # =================================================

    geometry_engine = GeometryEngine(
        lines=lines,
        polygons=polygons,
        frame_size=cfg["general_params"]["frame_size"],
    )

    # =================================================
    # Crossing Extractor
    # =================================================

    crossing_extractor = CrossingExtractor(
        line_ids=geometry_engine._line_ids,
        ttl_seconds=cfg["general_params"]["crossing_ttl"]
    )

    return (
        geometry_engine,
        crossing_extractor,
    )

