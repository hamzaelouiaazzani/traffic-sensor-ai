"""Geometry package organized into coordinate transformation and spatial processing."""

from traffic_analytics.geometry.coordinate_transformation import (
    CoordinateTransformError,
    Homography,
    HomographyTransformer,
    IPlanarCoordinateTransformer,
    build_coordinate_transformer,
    build_coordinate_transformer_from_config,
    coordinate_transformer_config_available,
    compute_homography,
    load_calibration,
    register_coordinate_transformer_backend,
    save_calibration_yaml,
    validate_coordinate_transformer_contract,
)
from traffic_analytics.geometry.spatial_processing import (
    ImageRasterMaskPolygonMembership,
    IPolygonMembership,
    Line,
    Point,
    Polygon,
    RayCastingPolygonMembership,
    SpatialProcessor,
)
from traffic_analytics.geometry.topology import Area

__all__ = [
    "Area",
    "CoordinateTransformError",
    "Homography",
    "HomographyTransformer",
    "ImageRasterMaskPolygonMembership",
    "IPlanarCoordinateTransformer",
    "IPolygonMembership",
    "Line",
    "Point",
    "Polygon",
    "RayCastingPolygonMembership",
    "SpatialProcessor",
    "build_coordinate_transformer",
    "build_coordinate_transformer_from_config",
    "coordinate_transformer_config_available",
    "compute_homography",
    "load_calibration",
    "register_coordinate_transformer_backend",
    "save_calibration_yaml",
    "validate_coordinate_transformer_contract",
]
