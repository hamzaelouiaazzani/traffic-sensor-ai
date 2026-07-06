import argparse
import logging
import signal
import threading
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional

from communication.mqtt_client import MqttClientConfig, SmartSensorMqttClient
from communication.services import ConfigurationReceiverService, SnapshotProviderService
from communication.topics import SensorTopics
from sensor_pipeline import run_sensor


class SensorState(str, Enum):
    BOOTING = "BOOTING"
    READY = "READY"
    PROCESSING = "PROCESSING"


CommandHandler = Callable[[bytes], None]


class SensorDaemon:
    """MQTT-controlled orchestrator for the smart traffic sensor lifecycle."""

    def __init__(
        self,
        source: str,
        broker_host: str,
        broker_port: int = 1883,
        mqtt_client_id: Optional[str] = None,
        sensor_id: str = "camera_1",
        config_path: str = "config/traffic_metrics.yaml",
        fps: Optional[float] = None,
        period_mins: float = 5.0,
        offline_queue_dir: str = "outputs/pending_mqtt",
        topic_root: str = "sensors",
    ):
        self.source = source
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.mqtt_client_id = mqtt_client_id or sensor_id
        self.sensor_id = sensor_id
        self.config_path = config_path
        self.fps = fps
        self.period_mins = period_mins
        self.offline_queue_dir = offline_queue_dir
        self.topic_root = topic_root

        self.mqtt_client = None
        self.topics = None
        self.snapshot_service = None
        self.configuration_receiver = None
        self.command_handlers: Dict[str, CommandHandler] = {}

        self.state = SensorState.BOOTING
        self.configured = Path(self.config_path).is_file()
        self.processing_thread = None
        self.stop_event = None
        self.shutdown_event = threading.Event()
        self._lock = threading.RLock()
        self._reboot_thread = None
        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        self.transition_to(SensorState.BOOTING)
        self._initialize_lifecycle()
        self.transition_to(SensorState.READY)
        self.run()

    def run(self) -> None:
        try:
            self.shutdown_event.wait()
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received; shutting down daemon.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.shutdown_event.set()
        self._stop_processing(wait=True)

        if self.mqtt_client is not None:
            self.mqtt_client.disconnect()
            self.mqtt_client = None

    def on_message(self, topic: str, payload: bytes) -> None:
        handler = self.command_handlers.get(topic)
        if handler is None:
            self.logger.warning("Ignoring unsupported command topic: %s", topic)
            return

        try:
            handler(payload)
        except Exception:
            self.logger.exception("Command handler failed for topic: %s", topic)

    def handle_snapshot_request(self, payload: bytes) -> None:
        if self.snapshot_service is None:
            self.logger.warning("Snapshot service is not initialized.")
            return

        self.snapshot_service.handle_snapshot_request(payload)

    def handle_configuration_command(self, payload: bytes) -> None:
        if self.configuration_receiver is None:
            self.logger.warning("Configuration receiver is not initialized.")
            return

        path = self.configuration_receiver.save_configuration_file(payload)
        with self._lock:
            self.configured = True

        self.logger.info("Configuration stored at %s", path)

    def handle_start_command(self, payload: bytes) -> None:
        with self._lock:
            if self.state is not SensorState.READY:
                self.logger.warning("START rejected while daemon is %s.", self.state.value)
                return

            if not self.configured:
                self.logger.warning("START rejected because no configuration has been received.")
                return

            self.stop_event = threading.Event()
            self.processing_thread = threading.Thread(
                target=self._run_processing,
                name=f"{self.sensor_id}-processing",
            )
            self.transition_to(SensorState.PROCESSING)
            self.processing_thread.start()

    def handle_stop_command(self, payload: bytes) -> None:
        with self._lock:
            if self.state is not SensorState.PROCESSING:
                self.logger.warning("STOP ignored while daemon is %s.", self.state.value)
                return

            if self.stop_event is not None:
                self.stop_event.set()

    def handle_reboot_command(self, payload: bytes) -> None:
        with self._lock:
            if self._reboot_thread is not None and self._reboot_thread.is_alive():
                self.logger.warning("REBOOT ignored because a reboot is already in progress.")
                return

            self._reboot_thread = threading.Thread(
                target=self.restart_daemon,
                name=f"{self.sensor_id}-restart",
            )
            self._reboot_thread.start()

    def transition_to(self, state: SensorState) -> None:
        with self._lock:
            if self.state is state:
                return

            self.logger.info("State transition: %s -> %s", self.state.value, state.value)
            self.state = state

    def _initialize_lifecycle(self) -> None:
        self.topics = SensorTopics(sensor_id=self.sensor_id, root=self.topic_root)
        self.configured = Path(self.config_path).is_file()

        self._create_mqtt_client()
        self._create_services()
        self._register_handlers()

        self.mqtt_client.connect()
        self.mqtt_client.subscribe(self.topics.all_commands)

    def _create_mqtt_client(self) -> None:
        self.mqtt_client = SmartSensorMqttClient(
            MqttClientConfig(
                broker_host=self.broker_host,
                broker_port=self.broker_port,
                client_id=self.mqtt_client_id,
                offline_queue_dir=str(Path(self.offline_queue_dir) / self.sensor_id),
            ),
            on_message=self.on_message,
        )

    def _create_services(self) -> None:
        config_path = Path(self.config_path)
        self.snapshot_service = SnapshotProviderService(
            mqtt_client=self.mqtt_client,
            topics=self.topics,
            camera_source=self.source,
        )
        self.configuration_receiver = ConfigurationReceiverService(
            target_dir=str(config_path.parent),
            filename=config_path.name,
        )

    def _register_handlers(self) -> None:
        self.command_handlers = {
            self.topics.snapshot_command: self.handle_snapshot_request,
            self.topics.configuration_command: self.handle_configuration_command,
            self.topics.start_command: self.handle_start_command,
            self.topics.stop_command: self.handle_stop_command,
            self.topics.reboot_command: self.handle_reboot_command,
        }

    def _run_processing(self) -> None:
        try:
            run_sensor(
                source=self.source,
                config_path=self.config_path,
                fps=self.fps,
                period_mins=self.period_mins,
                sensor_id=self.sensor_id,
                mqtt_client=self.mqtt_client,
                stop_event=self.stop_event,
            )
        except Exception:
            self.logger.exception("Processing pipeline terminated with an error.")
        finally:
            with self._lock:
                self.processing_thread = None
                self.stop_event = None
                if self.state is SensorState.PROCESSING:
                    self.transition_to(SensorState.READY)

    def _stop_processing(self, wait: bool) -> None:
        with self._lock:
            if self.stop_event is not None:
                self.stop_event.set()
            thread = self.processing_thread

        if wait and thread is not None and thread.is_alive():
            thread.join()

    def restart_daemon(self) -> None:
        self._stop_processing(wait=True)
        self.transition_to(SensorState.BOOTING)

        if self.mqtt_client is not None:
            self.mqtt_client.disconnect()

        self.mqtt_client = None
        self.topics = None
        self.snapshot_service = None
        self.configuration_receiver = None
        self.command_handlers = {}
        self.configured = Path(self.config_path).is_file()

        if self.shutdown_event.is_set():
            return

        self._initialize_lifecycle()
        self.transition_to(SensorState.READY)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the MQTT-controlled smart traffic sensor daemon."
    )
    parser.add_argument("--source", "--video", dest="source", required=True)
    parser.add_argument("--config", default="config/traffic_metrics.yaml")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--period-mins", type=float, default=5.0)
    parser.add_argument("--sensor-id", default="camera_1")
    parser.add_argument("--topic-root", default="sensors")
    parser.add_argument("--mqtt-broker-host", required=True)
    parser.add_argument("--mqtt-broker-port", type=int, default=1883)
    parser.add_argument("--mqtt-client-id", default=None)
    parser.add_argument("--mqtt-offline-queue-dir", default="outputs/pending_mqtt")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    daemon = SensorDaemon(
        source=args.source,
        broker_host=args.mqtt_broker_host,
        broker_port=args.mqtt_broker_port,
        mqtt_client_id=args.mqtt_client_id,
        sensor_id=args.sensor_id,
        config_path=args.config,
        fps=args.fps,
        period_mins=args.period_mins,
        offline_queue_dir=args.mqtt_offline_queue_dir,
        topic_root=args.topic_root,
    )

    signal.signal(signal.SIGTERM, lambda signum, frame: daemon.shutdown_event.set())
    daemon.start()


if __name__ == "__main__":
    main()
