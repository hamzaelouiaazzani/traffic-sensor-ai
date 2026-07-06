from dataclasses import dataclass


@dataclass(frozen=True)
class SensorTopics:
    sensor_id: str
    root: str = "sensors"

    @property
    def base(self) -> str:
        return f"{self.root}/{self.sensor_id}"

    @property
    def status(self) -> str:
        return f"{self.base}/status"

    @property
    def metrics(self) -> str:
        return f"{self.base}/metrics"

    @property
    def snapshot_command(self) -> str:
        return f"{self.base}/commands/snapshot"

    @property
    def snapshot_response(self) -> str:
        return f"{self.base}/responses/snapshot"

    @property
    def configuration_command(self) -> str:
        return f"{self.base}/commands/configuration"

    @property
    def configuration_response(self) -> str:
        return f"{self.base}/responses/configuration"

    @property
    def start_command(self) -> str:
        return f"{self.base}/commands/start"

    @property
    def start_response(self) -> str:
        return f"{self.base}/responses/start"

    @property
    def stop_command(self) -> str:
        return f"{self.base}/commands/stop"

    @property
    def stop_response(self) -> str:
        return f"{self.base}/responses/stop"

    @property
    def reboot_command(self) -> str:
        return f"{self.base}/commands/reboot"

    @property
    def reboot_response(self) -> str:
        return f"{self.base}/responses/reboot"

    @property
    def all_commands(self) -> str:
        return f"{self.base}/commands/#"

    @classmethod
    def all_sensor_metrics(cls, root: str = "sensors") -> str:
        return f"{root}/+/metrics"

    @classmethod
    def all_sensor_status(cls, root: str = "sensors") -> str:
        return f"{root}/+/status"

    @classmethod
    def all_sensor_responses(cls, root: str = "sensors") -> str:
        return f"{root}/+/responses/#"
