# coding: utf-8

import os
from collections import deque
from sys import stderr

from loguru import logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): loaction to save log file
        distributed_rank(int): device rank when multi-gpu environment
        mode(str): log file write mode, `append` or `override`. default is `a`.
    Return:
        logger instance.
    """
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    format = f"[Rank #{distributed_rank}] | " + "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
    if distributed_rank > 0:
        logger.remove()
        logger.add(
            stderr,
            format=format,
            level="WARNING",
        )
    logger.add(
        save_file,
        format=format,
        filter="",
        level="INFO" if distributed_rank == 0 else "WARNING",
        enqueue=True,
    )

    return logger


class AvgMeter(object):
    def __init__(self, window_size=50):
        self.window_size = window_size
        self._value_deque = deque(maxlen=window_size)
        self._total_value = 0.0
        self._wdsum_value = 0.0
        self._count_deque = deque(maxlen=window_size)
        self._total_count = 0.0
        self._wdsum_count = 0.0

    def reset(self):
        self._value_deque.clear()
        self._total_value = 0.0
        self._wdsum_value = 0.0
        self._count_deque.clear()
        self._total_count = 0.0
        self._wdsum_count = 0.0

    def update(self, value, n=1):
        if len(self._value_deque) >= self.window_size:
            self._wdsum_value -= self._value_deque.popleft()
            self._wdsum_count -= self._count_deque.popleft()
        self._value_deque.append(value * n)
        self._total_value += value * n
        self._wdsum_value += value * n
        self._count_deque.append(n)
        self._total_count += n
        self._wdsum_count += n

    @property
    def avg(self):
        return self.global_avg

    @property
    def global_avg(self):
        return self._total_value / max(self._total_count, 1e-5)

    @property
    def window_avg(self):
        return self._wdsum_value / max(self._wdsum_count, 1e-5)
