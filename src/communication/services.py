from pathlib import Path
from communication.mqtt_client import SmartSensorMqttClient
from communication.payloads import build_metrics_payload
from communication.topics import SensorTopics

from dataclasses import dataclass
from typing import Optional
from datetime import datetime

import cv2
import json



class MetricsPublisherService:
    def __init__(
        self,
        mqtt_client: SmartSensorMqttClient,
        topics: SensorTopics,
    ):
        self.mqtt_client = mqtt_client
        self.topics = topics

    def publish_period_metrics(self, period_result: dict) -> bool:
        payload = build_metrics_payload(
            sensor_id=self.topics.sensor_id,
            period_result=period_result,
        )

        payload = json.dumps(payload)
        return self.mqtt_client.publish(
            topic=self.topics.metrics,
            payload=payload,
        )


class ConfigurationPublisherService:
    def __init__(
        self,
        mqtt_client: SmartSensorMqttClient,
        topics: SensorTopics,
    ):
        self.mqtt_client = mqtt_client
        self.topics = topics

    def publish_configuration_file(
        self,
        yaml_path: str = r"C:\Users\hamza\Programs\traffic_metrics.yaml",
    ) -> bool:

        yaml_path = Path(yaml_path)

        with yaml_path.open("rb") as f:
            file_bytes = f.read()

        return self.mqtt_client.publish(
            topic=self.topics.configuration_command,
            payload=file_bytes,
        )






class ConfigurationReceiverService:
    def __init__(
        self,
        target_dir: str = r"C:\Users\hamza\Programs\traffic-sensor-ai\config",
        filename: str = "traffic_metrics.yaml",
    ):
        self.target_dir = Path(target_dir)
        self.filename = filename

    def save_configuration_file(self, file_bytes: bytes) -> str:
        self.target_dir.mkdir(parents=True, exist_ok=True)

        target_path = self.target_dir / self.filename
        with target_path.open("wb") as f:
            f.write(file_bytes)

        return str(target_path)






from dataclasses import dataclass
from typing import Optional

import cv2





class SnapshotProviderService:
    def __init__(self, mqtt_client, topics, camera_source: int = 0, jpeg_quality: int = 90):
        self.mqtt_client = mqtt_client
        self.topics = topics
        self.camera_source = camera_source
        self.jpeg_quality = jpeg_quality

    def capture_frame(self):
        print("[Snapshot] Opening camera...")
    
        cap = cv2.VideoCapture(self.camera_source)
        print("[Snapshot] isOpened =", cap.isOpened())
    
        if not cap.isOpened():
            print(f"[Snapshot] Cannot open camera source {self.camera_source}")
            return None
    
        print("[Snapshot] Reading frame...")
        ok, frame = cap.read()
        print("[Snapshot] read() =", ok)
    
        cap.release()
        print("[Snapshot] Camera released")
    
        if not ok:
            print("[Snapshot] Failed to capture frame")
            return None
    
        return frame
        

    def handle_snapshot_request(self, payload: bytes) -> bool:
        print("[Snapshot] Request received")

        frame = self.capture_frame()
        if frame is None:
            return False

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            print("[Snapshot] JPEG encoding failed")
            return False

        jpeg_bytes = buffer.tobytes()

        return self.mqtt_client.publish(
            topic=self.topics.snapshot_response,
            payload=jpeg_bytes,
        )




class SnapshotReceiverService:
    def __init__(
        self,
        output_dir: str = r"C:\Users\hamza\Programs\traffic-sensor-ai\outputs\snapshots",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, image_bytes: bytes) -> str:
        filename = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")

        output_path = self.output_dir / filename

        with output_path.open("wb") as f:
            f.write(image_bytes)

        return str(output_path)
