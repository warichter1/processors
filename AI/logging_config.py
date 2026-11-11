"""Queue-backed daily log handlers shared by the web and workflow layers."""

import logging
import os
import queue
import threading
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler


_listeners = {}
_loggers = {}
_configuration_lock = threading.Lock()


def get_queued_logger(name, log_directory, filename, formatter):
    """Return a logger writing through one queued, daily-rotated file sink."""
    directory = os.path.abspath(log_directory)
    os.makedirs(directory, exist_ok=True)
    log_path = os.path.join(directory, filename)
    logger_key = (name, log_path)
    sink_key = (log_path, formatter)

    with _configuration_lock:
        existing_logger = _loggers.get(logger_key)
        if existing_logger is not None:
            return existing_logger

        listener = _listeners.get(sink_key)
        if listener is None:
            log_queue = queue.Queue()
            file_handler = TimedRotatingFileHandler(
                log_path,
                when="midnight",
                interval=1,
                backupCount=6,
                encoding="utf-8",
                delay=True,
                utc=True,
            )
            file_handler.setFormatter(logging.Formatter(formatter))
            listener = QueueListener(
                log_queue,
                file_handler,
                respect_handler_level=True,
            )
            listener.start()
            _listeners[sink_key] = (log_queue, listener)
        else:
            log_queue, listener = listener

        logger = logging.getLogger(f"{name}.{log_path}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(QueueHandler(log_queue))
        _loggers[logger_key] = logger
        return logger
