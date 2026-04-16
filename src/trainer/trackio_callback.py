import logging

from transformers import TrainerCallback

from trackio_utils import log_trackio_metrics

logger = logging.getLogger(__name__)


class TrackioLoggingCallback(TrainerCallback):
    def __init__(self):
        self._logging_failed = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return

        try:
            log_trackio_metrics(logs, step=state.global_step)
        except Exception as exc:
            if not self._logging_failed:
                logger.warning("Trackio logging failed: %s", exc)
                self._logging_failed = True
