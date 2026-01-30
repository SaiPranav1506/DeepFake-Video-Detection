"""Simple structured telemetry logger for agent events
Logs JSON-lines to a file under `logs/agent_actions` with timestamps.
"""
import os
import json
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TelemetryLogger:
    def __init__(self, log_path: str = 'logs/agent_actions/telemetry.log'):
        self.log_path = log_path
        dirpath = os.path.dirname(self.log_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

    def log_event(self, event: Dict):
        """Append a structured event (dict) as a JSON line with timestamp."""
        record = dict(event)
        record.setdefault('ts', time.time())
        try:
            with open(self.log_path, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + '\n')
            logger.debug('Telemetried event: %s', record.get('event', ''))
        except Exception as e:
            logger.exception('Failed to write telemetry: %s', e)
