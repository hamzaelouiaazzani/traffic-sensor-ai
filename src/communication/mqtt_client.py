import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import paho.mqtt.client as mqtt

from communication.offline_queue import OfflineMessageQueue


MessageHandler = Callable[[str, bytes], None]


@dataclass
class MqttClientConfig:
    broker_host: str
    broker_port: int = 1883
    client_id: str = "smart_sensor"
    username: Optional[str] = None
    password: Optional[str] = None
    keepalive: int = 60
    qos: int = 1
    retain: bool = False
    offline_queue_dir: str = "outputs/pending_mqtt"


class SmartSensorMqttClient:
    def __init__(
        self,
        config: MqttClientConfig,
        on_message: Optional[MessageHandler] = None,
    ):
        self.config = config
        self.on_message = on_message
        self.queue = OfflineMessageQueue(config.offline_queue_dir)
        self.connected = False
        self.logger = logging.getLogger(__name__)

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=config.client_id,
        )

        if config.username is not None:
            self.client.username_pw_set(
                username=config.username,
                password=config.password,
            )

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def connect(self) -> None:
        self.client.connect(
            self.config.broker_host,
            self.config.broker_port,
            self.config.keepalive,
        )
        self.client.loop_start()

    def disconnect(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def subscribe(self, topics: Iterable[str]) -> None:
        if isinstance(topics, str):
            topics = [topics]

        for topic in topics:
            self.client.subscribe(topic, qos=self.config.qos)



    def publish(
        self,
        topic: str,
        payload: bytes | str,
        qos: Optional[int] = None,
        retain: Optional[bool] = None,
        queue_on_failure: bool = True,
    ) -> bool:
        print(f"[MQTT] Publishing to '{topic}', connected={self.connected}, payload={len(payload) if isinstance(payload, bytes) else len(str(payload))} bytes")
        qos = self.config.qos if qos is None else qos
        retain = self.config.retain if retain is None else retain
    
        if not self.connected:
            if queue_on_failure:
                self.queue.enqueue(topic, payload, qos, retain)
            return False
    
        info = self.client.publish(
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
        )
    
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            if queue_on_failure:
                self.queue.enqueue(topic, payload, qos, retain)
            return False
    
        return True

    def flush_pending(self) -> int:
        sent = 0

        if not self.connected:
            return sent

        for path, message in self.queue.iter_messages():
            ok = self.publish(
                topic=message["topic"],
                payload=message["payload"],
                qos=message["qos"],
                retain=message["retain"],
                queue_on_failure=False,
            )

            if not ok:
                break

            self.queue.remove(path)
            sent += 1

        return sent

    def _on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties=None,
    ) -> None:
        is_failure = getattr(reason_code, "is_failure", None)

        if callable(is_failure):
            self.connected = not is_failure()
        elif is_failure is not None:
            self.connected = not is_failure
        else:
            self.connected = reason_code == 0

        if self.connected:
            self.flush_pending()
        else:
            self.logger.warning("MQTT connection failed: %s", reason_code)

    def _on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties=None,
    ) -> None:
        self.connected = False
        print("Disconnected! Reason:", reason_code)

    def _on_message(
        self,
        client,
        userdata,
        message,
    ) -> None:
        if self.on_message is None:
            return

        self.on_message(message.topic, message.payload)