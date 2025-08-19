# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import time
import math
import logging
import os
import sys
from functools import lru_cache
from datetime import timedelta

logger = logging.getLogger()


@lru_cache()
def get_is_torch_run() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def get_is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not get_is_torch_run()


@lru_cache()
def get_global_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


@lru_cache()
def get_local_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


@lru_cache()
def get_world_size() -> int:
    if get_is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1


@lru_cache()
def get_is_master() -> bool:
    return get_global_rank() == 0



# def init_logger():
#     # Check for log level from environment variable
#     log_level_str = os.environ.get("LOGLEVEL", "INFO").upper()
#     log_level = getattr(logging, log_level_str, logging.INFO)

#     logger.setLevel(log_level)
#     ch = logging.StreamHandler()
#     ch.setLevel(log_level)
#     formatter = logging.Formatter(
#         "[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     )
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     # suppress verbose torch.profiler logging
#     os.environ["KINETO_LOG_LEVEL"] = "5"



class LogFormatter(logging.Formatter):
    """
    Custom logger for distributed jobs, displaying rank
    and preserving indent from the custom prefix format.
    """

    def __init__(self):
        self.start_time = time.time()
        self.rank = get_global_rank()
        self.show_rank = not get_is_slurm_job()  # srun has --label

    def formatTime(self, record):
        subsecond, seconds = math.modf(record.created)
        curr_date = (
            time.strftime("%y-%m-%d %H:%M:%S", time.localtime(seconds))
            + f".{int(subsecond * 1_000_000):06d}"
        )
        delta = timedelta(seconds=round(record.created - self.start_time))
        return f"{curr_date} - {delta}"

    def formatPrefix(self, record):
        fmt_time = self.formatTime(record)
        if self.show_rank:
            return f"{self.rank}: {record.levelname:<7} {fmt_time} - "
        else:
            return f"{record.levelname:<7} {fmt_time} - "

    def formatMessage(self, record, indent: str):
        content = record.getMessage()
        content = content.replace("\n", "\n" + indent)
        # Exception handling as in the default formatter, albeit with indenting
        # according to our custom prefix
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            content = content + indent.join(
                [l + "\n" for l in record.exc_text.splitlines()]
            )
            if content[-1:] == "\n":
                content = content[:-1]
        if record.stack_info:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            stack_text = self.formatStack(record.stack_info)
            content = content + indent.join([l + "\n" for l in stack_text.splitlines()])
            if content[-1:] == "\n":
                content = content[:-1]

        return content

    def format(self, record):
        prefix = self.formatPrefix(record)
        indent = " " * len(prefix)
        content = self.formatMessage(record, indent)
        return prefix + content


def set_root_log_level(log_level: str):
    logger = logging.getLogger()
    level: int | str = log_level.upper()
    try:
        level = int(log_level)
    except ValueError:
        pass
    try:
        logger.setLevel(level)  # type: ignore
    except Exception:
        logger.warning(
            f"Failed to set logging level to {log_level}, using default 'NOTSET'"
        )
        logger.setLevel(logging.NOTSET)


def init_logger(
    log_file: str | None = None,
    *,
    name: str | None = None,
    level: str = "NOTSET",
):
    """
    Setup logging.

    Args:
        log_file: A file name to save file logs to.
        name: The name of the logger to configure, by default the root logger.
        level: The logging level to use.
    """
    set_root_log_level(level)
    logger = logging.getLogger(name)

    # stdout: everything
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.NOTSET)
    stdout_handler.setFormatter(LogFormatter())

    # stderr: warnings / errors and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(LogFormatter())

    # set stream handlers
    logger.handlers.clear()
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)

    if log_file is not None and get_global_rank() == 0:
        # build file handler
        file_handler = logging.FileHandler(log_file, "a")
        file_handler.setLevel(logging.NOTSET)
        file_handler.setFormatter(LogFormatter())
        # update logger
        logger = logging.getLogger()
        logger.addHandler(file_handler)