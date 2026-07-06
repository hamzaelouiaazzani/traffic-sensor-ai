import argparse

from communication.mqtt_client import MqttClientConfig, SmartSensorMqttClient
from sensor_pipeline import run_sensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run smart-sensor traffic metrics over consecutive video periods."
    )
    parser.add_argument("--source", "--video", dest="source", required=True)
    parser.add_argument("--config", default="config/traffic_metrics.yaml")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--period-mins", type=float, default=5.0)
    parser.add_argument("--sensor-id", default="camera_1")
    parser.add_argument("--mqtt-broker-host", default=None)
    parser.add_argument("--mqtt-broker-port", type=int, default=1883)
    parser.add_argument("--mqtt-client-id", default=None)
    parser.add_argument("--mqtt-offline-queue-dir", default="outputs/pending_mqtt")
    return parser.parse_args()


def main():
    args = parse_args()
    mqtt_client = None

    if args.mqtt_broker_host is not None:
        mqtt_client = SmartSensorMqttClient(
            MqttClientConfig(
                broker_host=args.mqtt_broker_host,
                broker_port=args.mqtt_broker_port,
                client_id=args.mqtt_client_id or args.sensor_id,
                offline_queue_dir=str(
                    f"{args.mqtt_offline_queue_dir}/{args.sensor_id}"
                ),
            )
        )

    try:
        if mqtt_client is not None:
            mqtt_client.connect()

        run_sensor(
            source=args.source,
            config_path=args.config,
            fps=args.fps,
            period_mins=args.period_mins,
            sensor_id=args.sensor_id,
            mqtt_client=mqtt_client,
        )
    finally:
        if mqtt_client is not None:
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
