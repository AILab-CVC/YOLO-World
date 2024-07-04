# Origin from https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/logger.py
# Copyright (c) Facebook, Inc. and its affiliates. Apache License 2.0

import atexit
import datetime
import functools
import logging
import os
import sys
import time
import torch
from collections import Counter
from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler
from iopath.common.file_io import PathManager as PathManagerBase
from tabulate import tabulate
from termcolor import colored
from typing import Optional

__all__ = ['setup_logger', 'log_first_n', 'log_every_n', 'log_every_n_seconds']

PathManager = PathManagerBase()
PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())


class _ColorfulFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop('root_name') + '.'
        self._abbrev_name = kwargs.pop('abbrev_name', '')
        if len(self._abbrev_name) > 0:
            self._abbrev_name = self._abbrev_name + '.'
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored('WARNING', 'red', attrs=['blink'])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored('ERROR', 'red', attrs=['blink', 'underline'])
        else:
            return log
        return prefix + ' ' + log


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(output=None, distributed_rank=0, *, color=True, name='yolow', abbrev_name=None):
    """
    Initialize the yolow logger and set its verbosity level to "DEBUG".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "yolow" to "f" and leave other
            modules unchanged.
    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = 'yw' if name == 'yolow' else name

    plain_formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S')
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored('[%(asctime)s %(name)s]: ', 'green') + '%(message)s',
                datefmt='%m/%d %H:%M:%S',
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith('.txt') or output.endswith('.log'):
            filename = output
        else:
            filename = os.path.join(output, 'log.txt')
        if distributed_rank > 0:
            filename = filename + '.rank{}'.format(distributed_rank)
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = PathManager.open(filename, 'a', buffering=1024 if '://' in filename else -1)
    atexit.register(io.close)
    return io


"""
Below are some other convenient logging methods.
They are mainly adopted from
https://github.com/abseil/abseil-py/blob/master/absl/logging/__init__.py
"""


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join('utils', 'logger.') not in code.co_filename:
            mod_name = frame.f_globals['__name__']
            if mod_name == '__main__':
                mod_name = 'yolow'
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


_LOG_COUNTER = Counter()
_LOG_TIMER = {}


def log_first_n(lvl, msg, n=1, *, name=None, key='caller'):
    """
    Log only for the first n times.
    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
        key (str or tuple[str]): the string(s) can be one of "caller" or
            "message", which defines how to identify duplicated logs.
            For example, if called with `n=1, key="caller"`, this function
            will only log the first call from the same caller, regardless of
            the message content.
            If called with `n=1, key="message"`, this function will log the
            same content only once, even if they are called from different places.
            If called with `n=1, key=("caller", "message")`, this function
            will not log only if the same caller has logged the same message before.
    """
    if isinstance(key, str):
        key = (key, )
    assert len(key) > 0

    caller_module, caller_key = _find_caller()
    hash_key = ()
    if 'caller' in key:
        hash_key = hash_key + caller_key
    if 'message' in key:
        hash_key = hash_key + (msg, )

    _LOG_COUNTER[hash_key] += 1
    if _LOG_COUNTER[hash_key] <= n:
        logging.getLogger(name or caller_module).log(lvl, msg)


def log_every_n(lvl, msg, n=1, *, name=None):
    """
    Log once per n times.
    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()
    _LOG_COUNTER[key] += 1
    if n == 1 or _LOG_COUNTER[key] % n == 1:
        logging.getLogger(name or caller_module).log(lvl, msg)


def log_every_n_seconds(lvl, msg, n=1, *, name=None):
    """
    Log no more than once per n seconds.
    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()
    last_logged = _LOG_TIMER.get(key, None)
    current_time = time.time()
    if last_logged is None or current_time - last_logged >= n:
        logging.getLogger(name or caller_module).log(lvl, msg)
        _LOG_TIMER[key] = current_time


def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.
    Args:
        small_dict (dict): a result dictionary of only a few items.
    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt='pipe',
        floatfmt='.3f',
        stralign='center',
        numalign='center',
    )
    return table


def _log_api_usage(identifier: str):
    """
    Internal function used to log the usage of different yolow components
    inside facebook's infra.
    """
    torch._C._log_api_usage_once('yolow.' + identifier)


_CURRENT_STORAGE_STACK = []


def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert _CURRENT_STORAGE_STACK, "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]


class CommonMetricPrinter:
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.

    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, iters_per_epoch: int, max_iter: Optional[int] = None):
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
        """
        # self.logger = Logger(logfile="", level=logging.DEBUG)  #logging.getLogger(__name__)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self._max_iter = max_iter
        self._iters_per_epoch = iters_per_epoch
        self._last_write = None  # (step, time) of last call to write(). Used to compute ETA

    def _get_eta(self, storage) -> Optional[str]:
        if self._max_iter is None:
            return ''
        iteration = storage.iter
        try:
            eta_seconds = storage.history('time').median(1000) * (self._max_iter - iteration - 1)
            storage.put_scalar('eta_seconds', eta_seconds, smoothing_hint=False)
            return str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            # estimate eta on our own - more noisy
            eta_string = None
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (iteration - self._last_write[0])
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())
            return eta_string

    def write(self):
        import torch

        storage = get_event_storage()
        iteration = storage.iter
        epoch = (iteration + 0) // self._iters_per_epoch
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history('data_time').avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None
        try:
            iter_time = storage.history('time').global_avg()
        except KeyError:
            iter_time = None
        try:
            lr = '{:.5g}'.format(storage.history('lr').latest())
        except KeyError:
            lr = 'N/A'

        eta_string = self._get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(' {eta} epoch: {epoch}  iter: {iter}  {losses}  {time}{data_time}lr: {lr}  {memory}'.format(
            eta=f'eta: {eta_string} ' if eta_string else '',
            epoch=epoch,
            iter=iteration,  # median(20)
            losses='  '.join([
                '{}: {:.6f}'.format(k, v.latest()) for k, v in storage.histories().items()
                if ('loss' in k) or ('Cider' in k) or ('RewardCriterion' in k)
            ]),
            time='time: {:.4f}  '.format(iter_time) if iter_time is not None else '',
            data_time='data_time: {:.4f}  '.format(data_time) if data_time is not None else '',
            lr=lr,
            memory='max_mem: {:.0f}M'.format(max_mem_mb) if max_mem_mb is not None else '',
        ))

    def close(self):
        pass
