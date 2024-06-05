from supervisor import TTL
from typing import Optional

from .history import RemoteException
import logging

logger = logging.getLogger(__name__)

class Future:
    def __init__(self, ctrl: '_Controller'):
        self._ctrl = ctrl
        self._status = 'incomplete'
        self._callbacks = None
        self._result = None

    def _set_result(self, r):
        assert self._status == 'incomplete'
        self._result = r
        self._status = 'exception' if isinstance(r, RemoteException) else 'complete'
        if self._callbacks:
            for cb in self._callbacks:
                try:
                    cb(self)
                except Exception:
                    logger.exception("exception in controller's Future callback")
        self._callbacks = None
        self._ctrl = None

    def _wait(self, timeout: Optional[float]):
        if self._status != 'incomplete':
            return True
        # see if the future is done already
        # and we just haven't processed the messages
        self._ctrl._read_messages(0)
        if self._status != 'incomplete':
            return True
        # we might need to ask for status updates
        # from workers to be sure they have finished
        # enough work to count this future as finished.
        self._ctrl._request_status()
        if timeout is None:
            while self._status == 'incomplete':
                self._ctrl._read_messages(timeout=None)
        else:
            ttl = TTL(timeout)
            while self._status == 'incomplete':
                remaining = ttl()
                self._ctrl._read_messages(timeout=remaining)
                if remaining == 0:
                    return self._status != 'incomplete'
        return True

    def result(self, timeout: Optional[float] = None):
        if not self._wait(timeout):
            raise TimeoutError()
        assert self._result is not None
        if self._status == 'exception':
            raise self._result
        return self._result

    def done(self) -> bool:
        return self._wait(0)

    def exception(self, timeout: Optional[float] = None):
        if not self._wait(timeout):
            raise TimeoutError()
        return self._result if self._status == 'exception' else None

    def add_callback(self, callback):
        if not self._callbacks:
            self._callbacks = [callback]
        else:
            self._callbacks.append(callback)
