import json
from pathlib import Path
from typing import Dict, Iterable, Tuple
from uuid import uuid4


class OfflineMessageQueue:
    def __init__(self, queue_dir: str):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        topic: str,
        payload: Dict,
        qos: int,
        retain: bool,
    ) -> Path:
        path = self.queue_dir / f"{uuid4().hex}.json"

        message = {
            "topic": topic,
            "payload": payload,
            "qos": qos,
            "retain": retain,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(message, f, ensure_ascii=False)

        return path

    def iter_messages(self):
        for path in sorted(self.queue_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                message = json.load(f)
            yield path, message

    def remove(self, path: Path) -> None:
        path.unlink(missing_ok=True)
