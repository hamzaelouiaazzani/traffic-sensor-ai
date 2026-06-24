# detection/factory.py

from pathlib import Path

from detection.interface import DetectorError

from detection.ultralytics_detectors import (
    UltralyticsDetector
)

from detection.torchvision_detectors import (
    TorchvisionDetector
)


TORCHVISION_MODELS = {
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "retinanet_resnet50_fpn",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
    "fcos_resnet50_fpn",
}


def build_detector(
    model_name: str,
    **kwargs
):

    name = model_name.lower()

    # =================================================
    # Ultralytics models
    # =================================================

    if (
        "yolo" in name
        or "rtdetr" in name
    ):

        model_path = (
            Path("models")
            / f"{model_name}.pt"
        )

        return UltralyticsDetector(
            model_name=str(model_path),
            **kwargs
        )

    # =================================================
    # Torchvision models
    # =================================================

    elif name in TORCHVISION_MODELS:

        return TorchvisionDetector(
            model_name=model_name,
            **kwargs
        )

    # =================================================
    # Unsupported
    # =================================================

    raise DetectorError(
        f"Unsupported detector: {model_name}"
    )