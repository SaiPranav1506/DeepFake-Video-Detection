"""Active learning utilities for agent-driven labeling and retraining triggers

Provides a minimal queue for samples the agent abstains on and a simple
API to request labels (via a label provider callable) and flush them to
a labeled file for downstream retraining.
"""
import os
import json
import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ActiveLearner:
    def __init__(
        self,
        queue_path: str = 'data/active_queue.jsonl',
        labeled_path: str = 'data/active_labels.jsonl',
        retrain_threshold: int = 10,
        telemetry: Optional[object] = None,
    ):
        self.queue_path = queue_path
        self.labeled_path = labeled_path
        self.retrain_threshold = retrain_threshold
        self.telemetry = telemetry

        # Ensure directories exist
        for p in [self.queue_path, self.labeled_path]:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    def queue_for_label(self, prediction: Dict):
        """Add an abstained/low-confidence prediction to the labeling queue.

        `prediction` should be a serializable dict (e.g., from EnsemblePrediction.__dict__)
        """
        try:
            with open(self.queue_path, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(prediction, ensure_ascii=False) + '\n')

            if self.telemetry:
                self.telemetry.log_event({
                    'event': 'queued_for_label',
                    'video_id': prediction.get('video_id'),
                    'ensemble_prob': prediction.get('ensemble_prob'),
                    'confidence': prediction.get('confidence'),
                    'uncertainty': prediction.get('uncertainty')
                })

            logger.info('Queued for labeling: %s', prediction.get('video_id'))
        except Exception as e:
            logger.exception('Failed to queue for label: %s', e)

    def process_queue_with_label_provider(self, label_provider: Callable[[str], int]) -> int:
        """Consume the queue and apply `label_provider(video_id) -> label`.

        Writes labeled examples to `self.labeled_path` and returns number labeled.
        """
        if not os.path.exists(self.queue_path):
            return 0

        labeled_count = 0
        remaining = []

        try:
            with open(self.queue_path, 'r', encoding='utf-8') as fh:
                lines = fh.readlines()

            for line in lines:
                try:
                    rec = json.loads(line)
                    vid = rec.get('video_id')
                    label = label_provider(vid)
                    labeled_record = dict(rec)
                    labeled_record['label'] = int(label)
                    with open(self.labeled_path, 'a', encoding='utf-8') as lf:
                        lf.write(json.dumps(labeled_record, ensure_ascii=False) + '\n')

                    labeled_count += 1
                    if self.telemetry:
                        self.telemetry.log_event({
                            'event': 'labeled',
                            'video_id': vid,
                            'label': int(label)
                        })
                except Exception:
                    # keep the problematic line for later
                    remaining.append(line)

            # rewrite queue with any remaining
            with open(self.queue_path, 'w', encoding='utf-8') as qf:
                qf.writelines(remaining)

            logger.info('Processed queue, labeled %d samples', labeled_count)
        except Exception as e:
            logger.exception('Failed to process queue: %s', e)

        return labeled_count

    def labeled_count(self) -> int:
        if not os.path.exists(self.labeled_path):
            return 0
        try:
            with open(self.labeled_path, 'r', encoding='utf-8') as fh:
                return sum(1 for _ in fh)
        except Exception:
            return 0

    def should_trigger_retrain(self) -> bool:
        return self.labeled_count() >= self.retrain_threshold
